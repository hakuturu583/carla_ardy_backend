"""What the server needs of a motion generator, in the contract's terms.

:class:`MotionEngine` is the seam between the gRPC server
(:mod:`carla_ardy_backend.server`) and the model
(:class:`carla_ardy_backend.ardy_engine.ArdyEngine`): sessions opened with
their conditions, stepped from a kept frame on, closed.  Everything here is
in the contract frame (:mod:`carla_ardy_backend.frames`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "EngineInfo",
    "Frame",
    "Keyframe",
    "MotionEngine",
    "SessionSpec",
    "Span",
    "Waypoint",
]

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class EngineInfo:
    model: str
    fps: float
    horizon_frames: int
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class Span:
    start_frame: int
    text: str


@dataclass(frozen=True)
class Waypoint:
    frame: int
    x: float
    y: float
    heading: float | None = None


@dataclass(frozen=True)
class Keyframe:
    frame: int
    joints: tuple[Vec3, ...]


@dataclass(frozen=True)
class SessionSpec:
    root: tuple[float, float]
    heading: float
    seed: int = 0
    texts: tuple[Span, ...] = ()
    waypoints: tuple[Waypoint, ...] = ()
    keyframes: tuple[Keyframe, ...] = field(default_factory=tuple)

    def text_at(self, frame: int) -> str:
        """The text in force at *frame* (``""`` before the first span)."""
        text = ""
        for span in sorted(self.texts, key=lambda s: s.start_frame):
            if span.start_frame <= frame:
                text = span.text
        return text


@dataclass(frozen=True)
class Frame:
    frame: int
    root: Vec3
    heading: float
    joints: tuple[Vec3, ...]


class MotionEngine(Protocol):
    def info(self) -> EngineInfo: ...

    def open(self, session_id: str, spec: SessionSpec) -> None: ...

    def step(self, session_id: str, keep_until_frame: int) -> list[Frame]:
        """The frames after the history kept, at least ``keep_until_frame + 1``
        on.  Raises :class:`KeyError` for a session that is not open."""
        ...

    def close(self, session_id: str) -> None: ...
