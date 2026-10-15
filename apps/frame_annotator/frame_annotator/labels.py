"""Per-frame labels for one clip.

Every frame is labelled individually (never the clip as a whole). A frame holds
an obstacle label, a deviation label and an ``unusable`` flag. Instances are
immutable: ``apply`` returns a new ``ClipAnnotation``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

OBSTACLES = ("free_path", "obstacle")
DEVIATIONS = ("left", "right", "none")
NO_SIDE = "none"
FREE_PATH = "free_path"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def is_safe_id(value) -> bool:
    """Clip and annotator ids are used in file paths and URLs: keep them tame."""
    return isinstance(value, str) and _ID_PATTERN.match(value) is not None


@dataclass(frozen=True)
class FrameLabel:
    obstacle: Optional[str]
    deviation: Optional[str]
    unusable: bool

    @property
    def is_done(self) -> bool:
        """A frame is done when it is flagged unusable or fully labelled."""
        return self.unusable or (self.obstacle is not None and self.deviation is not None)


_EMPTY = FrameLabel(None, None, False)


def _check_choice(value, allowed, name):
    if value is not None and value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


@dataclass(frozen=True)
class ClipAnnotation:
    frame_count: int
    _labels: tuple  # Optional[FrameLabel] per frame; None = untouched

    @classmethod
    def empty(cls, frame_count: int) -> "ClipAnnotation":
        if frame_count < 0:
            raise ValueError("frame_count must be >= 0")
        return cls(frame_count, (None,) * frame_count)

    def label_at(self, index: int) -> Optional[FrameLabel]:
        return self._labels[index]

    @property
    def labelled_count(self) -> int:
        return sum(1 for label in self._labels if label is not None and label.is_done)

    @property
    def is_complete(self) -> bool:
        return self.frame_count > 0 and self.labelled_count == self.frame_count

    def apply(
        self,
        start: int,
        end: int,
        *,
        obstacle: Optional[str] = None,
        deviation: Optional[str] = None,
        unusable: Optional[bool] = None,
    ) -> "ClipAnnotation":
        """Return a copy with ``obstacle``/``deviation``/``unusable`` set on
        frames ``start..end`` inclusive. ``None`` leaves that field unchanged."""
        if not 0 <= start <= end < self.frame_count:
            raise ValueError(f"invalid frame range {start}..{end} for {self.frame_count} frames")
        _check_choice(obstacle, OBSTACLES, "obstacle")
        _check_choice(deviation, DEVIATIONS, "deviation")
        if unusable is not None and not isinstance(unusable, bool):
            raise ValueError("unusable must be a bool")
        if unusable and (obstacle is not None or deviation is not None):
            raise ValueError("cannot label and flag unusable in the same call")
        if obstacle == FREE_PATH and deviation not in (None, NO_SIDE):
            raise ValueError("free_path cannot be combined with a left/right deviation")

        labels = list(self._labels)
        for i in range(start, end + 1):
            labels[i] = self._next_label(labels[i], obstacle, deviation, unusable)
        return ClipAnnotation(self.frame_count, tuple(labels))

    @staticmethod
    def _next_label(current, obstacle, deviation, unusable):
        if unusable:
            return FrameLabel(None, None, True)
        cur = current or _EMPTY
        if cur.unusable:
            if obstacle is None and deviation is None:
                return None if unusable is False else current
            cur = _EMPTY  # labelling an unusable frame makes it usable again
        new_obstacle = cur.obstacle if obstacle is None else obstacle
        new_deviation = cur.deviation if deviation is None else deviation
        if obstacle == FREE_PATH:
            new_deviation = NO_SIDE
        elif obstacle is not None and cur.obstacle == FREE_PATH and deviation is None:
            new_deviation = None  # 'none' was forced by free_path, not chosen
        if new_obstacle == FREE_PATH and new_deviation not in (None, NO_SIDE):
            raise ValueError("a free_path frame cannot take a left/right deviation")
        if new_obstacle is None and new_deviation is None:
            return None
        return FrameLabel(new_obstacle, new_deviation, False)

    def to_runs(self) -> list:
        """Run-length rows ``[start, end, obstacle, deviation, unusable]``."""
        runs: list = []
        for i, label in enumerate(self._labels):
            if label is None:
                continue
            row = [label.obstacle, label.deviation, label.unusable]
            if runs and runs[-1][1] == i - 1 and runs[-1][2:] == row:
                runs[-1][1] = i
            else:
                runs.append([i, i, *row])
        return runs

    @classmethod
    def from_runs(cls, frame_count: int, runs) -> "ClipAnnotation":
        labels: list = [None] * frame_count
        for run in runs:
            try:
                start, end, obstacle, deviation, unusable = run
            except (TypeError, ValueError):
                raise ValueError(f"malformed run {run!r}") from None
            if not (isinstance(start, int) and isinstance(end, int)):
                raise ValueError(f"run bounds must be integers: {run!r}")
            if not 0 <= start <= end < frame_count:
                raise ValueError(f"run {start}..{end} outside {frame_count} frames")
            if not isinstance(unusable, bool):
                raise ValueError(f"run unusable flag must be a bool: {run!r}")
            _check_choice(obstacle, OBSTACLES, "obstacle")
            _check_choice(deviation, DEVIATIONS, "deviation")
            if unusable and (obstacle is not None or deviation is not None):
                raise ValueError("an unusable frame cannot carry labels")
            if obstacle == FREE_PATH and deviation not in (None, NO_SIDE):
                raise ValueError("free_path frames cannot have a left/right deviation")
            label = FrameLabel(obstacle, deviation, unusable)
            for i in range(start, end + 1):
                if labels[i] is not None:
                    raise ValueError(f"overlapping runs at frame {i}")
                labels[i] = label
        return cls(frame_count, tuple(labels))
