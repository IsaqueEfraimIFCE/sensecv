"""Timed events for one full clip.

The annotator watches the whole recording and marks *when* each event happens
and *what kind* it is, using the vocabulary of the SenseCV viewer's crop form:

* ``kind``: ``desvio`` (lateral deviation) or ``parada`` (stop);
* ``side``: ``esquerda`` / ``direita`` - where the person goes, required for a
  desvio and absent for a parada;
* ``obstacle``: ``centro`` / ``esquerda`` / ``direita`` / ``nao_visivel`` -
  where the obstacle sits in the path (optional);
* ``start_s`` (T1, the body starts to react) and ``end_s`` (T2, the trajectory
  is stable again / the person is fully stopped). ``end_s`` is required for a
  desvio and optional for a parada.

A clip is closed with a verdict: ``eventos`` (at least one event),
``sem_evento`` (free walk) or ``inutilizavel``. Instances are immutable.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

KINDS = ("desvio", "parada")
SIDES = ("esquerda", "direita")
OBSTACLES = ("centro", "esquerda", "direita", "nao_visivel")
VERDICTS = ("eventos", "sem_evento", "inutilizavel")
WITH_EVENTS = "eventos"
DESVIO = "desvio"
MAX_EVENTS = 20
MAX_NOTE_LEN = 500
# The browser may report a duration a few frames longer than the metadata.
DURATION_SLACK_S = 0.5
TIME_DECIMALS = 3

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def is_safe_id(value) -> bool:
    """Clip and annotator ids are used in file paths and URLs: keep them tame."""
    return isinstance(value, str) and _ID_PATTERN.fullmatch(value) is not None


def _check_choice(value, allowed, name, *, optional=False):
    if value is None and optional:
        return
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _time(value, name: str, limit: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number of seconds")
    if not 0 <= value <= limit:
        raise ValueError(f"{name} {value} outside 0..{limit:.3f} s")
    return round(float(value), TIME_DECIMALS)


def frame_at(seconds: float, fps: float) -> int:
    """Frame shown at ``seconds`` (``floor(t * fps)``, as in the SenseCV viewer)."""
    return int(math.floor(seconds * fps + 1e-6))


@dataclass(frozen=True)
class Event:
    kind: str
    side: Optional[str]
    obstacle: Optional[str]
    start_s: float
    end_s: Optional[float]

    @property
    def last_s(self) -> float:
        return self.start_s if self.end_s is None else self.end_s

    @classmethod
    def parse(cls, raw, limit: float) -> "Event":
        if not isinstance(raw, dict):
            raise ValueError("an event must be an object")
        kind = raw.get("kind")
        _check_choice(kind, KINDS, "kind")
        side = raw.get("side")
        if kind == DESVIO:
            _check_choice(side, SIDES, "side")
        elif side is not None:
            raise ValueError("a parada has no side")
        obstacle = raw.get("obstacle")
        _check_choice(obstacle, OBSTACLES, "obstacle", optional=True)
        start = _time(raw.get("start_s"), "start_s", limit)
        end = raw.get("end_s")
        if end is None:
            if kind == DESVIO:
                raise ValueError("a desvio needs an end time")
        else:
            end = _time(end, "end_s", limit)
            if end <= start:
                raise ValueError("end_s must be after start_s")
        return cls(kind, side, obstacle, start, end)

    def to_json(self, fps: float) -> dict:
        return {
            "kind": self.kind, "side": self.side, "obstacle": self.obstacle,
            "start_s": self.start_s, "end_s": self.end_s,
            "start_frame": frame_at(self.start_s, fps),
            "end_frame": None if self.end_s is None else frame_at(self.end_s, fps),
        }


@dataclass(frozen=True)
class ClipEvents:
    duration_s: float
    fps: float
    events: tuple = ()
    verdict: Optional[str] = None
    done: bool = False
    note: str = ""

    @classmethod
    def empty(cls, duration_s: float, fps: float) -> "ClipEvents":
        if duration_s <= 0 or fps <= 0:
            raise ValueError("duration_s and fps must be positive")
        return cls(duration_s, fps)

    @property
    def status(self) -> str:
        """``pendente`` (untouched), ``andamento`` (started) or ``concluido``."""
        if self.done:
            return "concluido"
        if self.events or self.verdict or self.note:
            return "andamento"
        return "pendente"

    def replace_with(self, payload, *, limit_s: Optional[float] = None) -> "ClipEvents":
        """Validated copy holding ``payload`` (``events``, ``verdict``, ``done``,
        ``note``). Events are sorted by start time and may not overlap. Times
        must lie within the clip (plus slack) unless ``limit_s`` widens it."""
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        raw_events = payload.get("events", [])
        if not isinstance(raw_events, list) or len(raw_events) > MAX_EVENTS:
            raise ValueError(f"events must be a list of at most {MAX_EVENTS}")
        limit = self.duration_s + DURATION_SLACK_S if limit_s is None else limit_s
        events = tuple(sorted((Event.parse(e, limit) for e in raw_events),
                              key=lambda e: (e.start_s, e.last_s)))
        for before, after in zip(events, events[1:]):
            if after.start_s <= before.last_s:
                raise ValueError("events overlap; adjust their start and end times")
        verdict = payload.get("verdict")
        _check_choice(verdict, VERDICTS, "verdict", optional=True)
        if events and verdict not in (None, WITH_EVENTS):
            raise ValueError(f"a clip with events cannot be {verdict!r}")
        if events:
            verdict = WITH_EVENTS
        elif verdict == WITH_EVENTS:
            verdict = None  # every event was deleted
        done = payload.get("done", False)
        if not isinstance(done, bool):
            raise ValueError("done must be a bool")
        if done and verdict is None:
            raise ValueError("mark at least one event, or choose sem_evento / inutilizavel")
        note = payload.get("note", "")
        if not isinstance(note, str) or len(note) > MAX_NOTE_LEN:
            raise ValueError(f"note must be text of at most {MAX_NOTE_LEN} characters")
        return ClipEvents(self.duration_s, self.fps, events, verdict, done, note.strip())

    def to_json(self) -> dict:
        return {
            "events": [e.to_json(self.fps) for e in self.events],
            "verdict": self.verdict,
            "done": self.done,
            "note": self.note,
            "status": self.status,
        }

    def summary(self) -> dict:
        return {"status": self.status, "events": len(self.events), "verdict": self.verdict}

    @classmethod
    def from_json(cls, data, duration_s: float, fps: float) -> "ClipEvents":
        """Inverse of ``to_json``; derived fields (frames, status) are recomputed.

        Lenient about the clip length: a re-published index may report a shorter
        duration than the one the times were marked against, and the annotator
        must still be able to open the clip and fix it (``replace_with`` stays
        strict, so the next save asks for in-range times)."""
        if not isinstance(data, dict):
            raise ValueError("stored annotation must be an object")
        raw_events = data.get("events", [])
        if not isinstance(raw_events, list) or not all(isinstance(e, dict) for e in raw_events):
            raise ValueError("stored events are malformed")
        events = [{k: e.get(k) for k in ("kind", "side", "obstacle", "start_s", "end_s")}
                  for e in raw_events]
        return cls.empty(duration_s, fps).replace_with({**data, "events": events},
                                                       limit_s=math.inf)
