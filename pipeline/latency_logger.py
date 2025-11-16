from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
import contextlib
import time

import torch

try:
    import nvtx  # type: ignore
except Exception:
    nvtx = None

def _cuda_event():
    return torch.cuda.Event(enable_timing=True)

@dataclass
class LatencySample:
    """Holds timings for one frame."""
    frame_idx: int
    meta: Dict[str, Any] = field(default_factory=dict)
    # stage -> milliseconds
    stages_ms: Dict[str, float] = field(default_factory=dict)
    # step-level breakdowns for denoising (list aligned with scheduler steps)
    denoise_steps_ms: List[Dict[str, float]] = field(default_factory=list)
    # timestamps for inter-frame latency calculation
    start_time_ms: Optional[float] = None
    end_time_ms: Optional[float] = None
    inter_frame_gap_ms: Optional[float] = None  # gap from previous frame end to this frame start

@dataclass
class BatchSample:
    """Holds aggregate timings for one batch (sum over frames)."""
    batch_idx: int
    meta: Dict[str, Any] = field(default_factory=dict)
    stages_ms: Dict[str, float] = field(default_factory=dict)  # batch-level timers
    frames_sum_stages_ms: Dict[str, float] = field(default_factory=dict)  # sums across frames
    frames_count: int = 0
    frames_sum_denoise_steps_ms: List[Dict[str, float]] = field(default_factory=list)
    # Track inter-frame latencies
    inter_frame_gaps_ms: List[float] = field(default_factory=list)
    # Track prompt switches
    prompt_switches: List[Dict[str, Any]] = field(default_factory=list)

class LatencyLogger:
    """
    CUDA-event based latency logger with an easy with-block API.

    Frame timers:
        with logger.timer("vae_decode"): ...

    Step-scoped timers for the denoise loop:
        with logger.step_timer(step_idx) as step:
            with step.timer("denoise_forward"): ...
            with step.timer("scheduler_add_noise"): ...

    Batch support:
        with logger.batch_scope(batch_idx, batch_size=len(batch)):
            with logger.batch_timer("batch_total"): ...
            # generate frames here
    """
    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        nested_batches: bool = True,
        frames_also_jsonl: bool = False,
    ):
        self.jsonl_path = jsonl_path or "latency.jsonl"
        self.nested_batches = nested_batches
        self.frames_also_jsonl = frames_also_jsonl
        self._current: LatencySample = LatencySample(frame_idx=-1)
        self._active_events: Dict[str, Any] = {}
        self._active_ranges: List[str] = []
        # ----- batch context -----
        self._batch: BatchSample = BatchSample(batch_idx=0)
        self._batch_frames_buf: Optional[List[Dict[str, Any]]] = None  # buffer when nested
        # ----- inter-frame tracking -----
        self._last_frame_end_ms: Optional[float] = None

    # --------- frame lifecycle ----------
    def start_frame(self, frame_idx: int, **meta):
        merged_meta = dict(meta)
        if self._batch is not None:
            # inherit current batch meta (e.g., batch_idx, batch_size, dataloader meta)
            merged_meta = {**self._batch.meta, **merged_meta}
        self._current = LatencySample(frame_idx=frame_idx, meta=merged_meta)
        self._active_events.clear()
        self._active_ranges.clear()
        
        # Record start time for inter-frame latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._current.start_time_ms = time.perf_counter() * 1000.0
        
        # Calculate inter-frame gap
        if self._last_frame_end_ms is not None:
            self._current.inter_frame_gap_ms = self._current.start_time_ms - self._last_frame_end_ms

    def finalize_frame(self):
        if self._current is None:
            return

        # Record end time for inter-frame latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._current.end_time_ms = time.perf_counter() * 1000.0
        self._last_frame_end_ms = self._current.end_time_ms

        # Build the frame dict once
        frame_row = {
            "record": "frame",
            "frame_idx": self._current.frame_idx,
            "meta": self._current.meta,
            "stages_ms": self._current.stages_ms,
            "denoise_steps_ms": self._current.denoise_steps_ms,
            "inter_frame_gap_ms": self._current.inter_frame_gap_ms,
        }

        # If we're inside a batch and nested output is enabled, buffer it;
        # optionally also write per-frame JSONL lines right now.
        if self._batch is not None and self.nested_batches:
            if self._batch_frames_buf is None:
                self._batch_frames_buf = []
            self._batch_frames_buf.append(frame_row)
            if self.frames_also_jsonl:
                os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(frame_row) + "\n")
        else:
            # No active batch (or nested disabled): write the frame as a standalone line
            os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(frame_row) + "\n")

        # accumulate into batch aggregates if active
        if self._batch is not None:
            self._batch.frames_count += 1
            # sum frame stages
            for k, v in self._current.stages_ms.items():
                self._batch.frames_sum_stages_ms[k] = self._batch.frames_sum_stages_ms.get(k, 0.0) + float(v)
            # sum step breakdowns by index
            for i, step_dict in enumerate(self._current.denoise_steps_ms):
                while len(self._batch.frames_sum_denoise_steps_ms) <= i:
                    self._batch.frames_sum_denoise_steps_ms.append({})
                for sk, sv in step_dict.items():
                    agg = self._batch.frames_sum_denoise_steps_ms[i]
                    agg[sk] = agg.get(sk, 0.0) + float(sv)
            # track inter-frame gaps
            if self._current.inter_frame_gap_ms is not None:
                self._batch.inter_frame_gaps_ms.append(self._current.inter_frame_gap_ms)

        self._current = None
    
    def mark_prompt_switch(self, frame_idx: int, old_prompt: str, new_prompt: str):
        """Record a prompt switch event at the specified frame."""
        if self._batch is not None:
            switch_event = {
                "frame_idx": frame_idx,
                "old_prompt": old_prompt,
                "new_prompt": new_prompt,
                "timestamp_ms": time.perf_counter() * 1000.0
            }
            self._batch.prompt_switches.append(switch_event)

    # --------- simple stage timers ----------
    @contextlib.contextmanager
    def timer(self, name: str) -> Iterator[None]:
        if self._current is None:
            # no-op if not started
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = _cuda_event(); end = _cuda_event()
            start.record()
            if nvtx: self._push_nvtx(name)
            try:
                yield
            finally:
                if nvtx: self._pop_nvtx()
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end)
                # Check if current is still valid before accessing
                if self._current is not None:
                    self._current.stages_ms[name] = self._current.stages_ms.get(name, 0.0) + float(ms)
        else:
            t0 = time.perf_counter()
            if nvtx: self._push_nvtx(name)
            try:
                yield
            finally:
                if nvtx: self._pop_nvtx()
                ms = (time.perf_counter() - t0) * 1000.0
                # Check if current is still valid before accessing
                if self._current is not None:
                    self._current.stages_ms[name] = self._current.stages_ms.get(name, 0.0) + float(ms)

    # --------- denoise step-scoped timers ----------
    @contextlib.contextmanager
    def step_timer(self, step_idx: int):
        """Creates a scoped recorder for a single denoising step."""
        step_rec = _StepRecorder(self, step_idx)
        try:
            yield step_rec
        finally:
            step_rec.close()

    # --------- NVTX helpers ----------
    def _push_nvtx(self, name: str):
        if nvtx:
            nvtx.range_push(name)
            self._active_ranges.append(name)

    def _pop_nvtx(self):
        if nvtx and self._active_ranges:
            nvtx.range_pop()
            self._active_ranges.pop()

    # --------- batch lifecycle ----------
    def start_batch(self, batch_idx: int, **meta):
        """Begin a batch; all frames inherit this meta."""
        self._batch = BatchSample(batch_idx=batch_idx, meta={"batch_idx": batch_idx, **meta})
        if self.nested_batches:
            self._batch_frames_buf = []

    def finalize_batch(self):
        """Flush a batch summary row with aggregates (and nested frames if enabled)."""
        if self._batch is None:
            return
        os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
        batch_row: Dict[str, Any] = {
            "record": "batch",
            "batch_idx": self._batch.batch_idx,
            "meta": self._batch.meta,
            "batch_stages_ms": self._batch.stages_ms,
            "frames_count": self._batch.frames_count,
            "frames_sum_stages_ms": self._batch.frames_sum_stages_ms,
            "frames_sum_denoise_steps_ms": self._batch.frames_sum_denoise_steps_ms,
            "inter_frame_gaps_ms": self._batch.inter_frame_gaps_ms,
            "prompt_switches": self._batch.prompt_switches,
        }
        if self.nested_batches:
            batch_row["frames"] = self._batch_frames_buf or []
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(batch_row) + "\n")
        self._batch = None
        self._batch_frames_buf = None

    @contextlib.contextmanager
    def batch_scope(self, batch_idx: int, **meta):
        """Convenience wrapper: start_batch → yield → finalize_batch."""
        self.start_batch(batch_idx, **meta)
        try:
            yield self
        finally:
            self.finalize_batch()

    # --------- batch timers ----------
    @contextlib.contextmanager
    def batch_timer(self, name: str):
        """Batch-scoped timers (e.g., dataloader_next, h2d, d2h, batch_total)."""
        if self._batch is None:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = _cuda_event(); end = _cuda_event()
            start.record()
            if nvtx: self._push_nvtx(f"batch:{name}")
            try:
                yield
            finally:
                if nvtx: self._pop_nvtx()
                end.record(); torch.cuda.synchronize()
                ms = start.elapsed_time(end)
                # Check if batch is still valid before accessing
                if self._batch is not None:
                    self._batch.stages_ms[name] = self._batch.stages_ms.get(name, 0.0) + float(ms)
        else:
            t0 = time.perf_counter()
            if nvtx: self._push_nvtx(f"batch:{name}")
            try:
                yield
            finally:
                if nvtx: self._pop_nvtx()
                ms = (time.perf_counter() - t0) * 1000.0
                # Check if batch is still valid before accessing
                if self._batch is not None:
                    self._batch.stages_ms[name] = self._batch.stages_ms.get(name, 0.0) + float(ms)

class _StepRecorder:
    def __init__(self, parent: LatencyLogger, step_idx: int):
        self.parent = parent
        self.step_idx = step_idx
        self._buf: Dict[str, float] = {}
        # ensure list length
        cur = self.parent._current
        if cur is not None:
            while len(cur.denoise_steps_ms) <= step_idx:
                cur.denoise_steps_ms.append({})

    @contextlib.contextmanager
    def timer(self, name: str):
        if self.parent._current is None:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = _cuda_event(); end = _cuda_event()
            start.record()
            if nvtx: self.parent._push_nvtx(f"step:{self.step_idx}:{name}")
            try:
                yield
            finally:
                if nvtx: self.parent._pop_nvtx()
                end.record(); torch.cuda.synchronize()
                ms = start.elapsed_time(end)
                self._buf[name] = self._buf.get(name, 0.0) + float(ms)
        else:
            t0 = time.perf_counter()
            if nvtx: self.parent._push_nvtx(f"step:{self.step_idx}:{name}")
            try:
                yield
            finally:
                if nvtx: self.parent._pop_nvtx()
                ms = (time.perf_counter() - t0) * 1000.0
                self._buf[name] = self._buf.get(name, 0.0) + float(ms)

    def close(self):
        cur = self.parent._current
        if cur is not None:
            cur.denoise_steps_ms[self.step_idx].update(self._buf)
