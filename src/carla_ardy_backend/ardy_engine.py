"""ARDY as a :class:`~carla_ardy_backend.engine.MotionEngine`.

Each session keeps the motion generated so far, as ARDY's normalised
features, and its frame numbers from the session's start.  A step keeps the
frames up to the one it is told (rounded up to one token of ARDY's), takes
the last second or so of them as the history, and generates ARDY's horizon
on from there -- conditioned by the text in force where the new frames start
and by the waypoints and keyframes ahead, as the interactive demo of ARDY
does (``scripts/interactive_demo/generation.py``).

Every step draws from a seed of the session's seed and the frame it starts
from, so the same calls give the same motion.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .engine import EngineInfo, Frame, SessionSpec
from .frames import ardy_to_contract, contract_to_ardy

logger = logging.getLogger(__name__)

__all__ = ["ArdyEngine"]


@dataclass
class _Session:
    spec: SessionSpec
    #: Normalised features ``[1, T, D]`` of the frames kept, or None.
    motion: Any = None
    #: Their posed joints ``[1, T, J, 3]`` (ARDY's frame).
    joints: Any = None
    #: The session frame of ``motion[:, 0]``.
    first: int = 0


class ArdyEngine:
    """ARDY on one device.

    Args:
        model_name: The checkpoint (a registry nickname, ``soma``, or a name).
        device: The torch device.
        acceleration: ``none`` or ``compile`` (torch.compile the denoiser).
        history_frames: Frames of history a step is generated from (rounded
            down to whole tokens); ``None``: one second.
        diffusion_steps: Denoising steps; ``None``: the model's own.
        cfg_weight: Classifier-free guidance weights (text, constraints).
    """

    def __init__(
        self,
        model_name: str = "soma",
        device: str = "cuda",
        acceleration: str = "none",
        history_frames: int | None = None,
        diffusion_steps: int | None = None,
        cfg_weight: tuple[float, float] = (2.0, 2.0),
    ) -> None:
        import torch
        from ardy.model.load_model import load_model
        from ardy.tools import disable_tf32

        disable_tf32()
        self._torch = torch
        logger.info("loading ARDY %s on %s", model_name, device)
        model = load_model(model_name, device=device)
        if acceleration == "compile":
            model.compile_denoiser(backend="inductor", mode="default", dynamic=False)
            model.warmup()
        elif acceleration != "none":
            raise ValueError(f"acceleration is none or compile, not {acceleration!r}")
        self._model = model
        self._device = torch.device(device)
        self._rep = model.motion_rep
        self._skeleton = self._rep.skeleton
        self._model_name = model_name
        self._fps = float(self._rep.fps)
        self._patch = int(model.num_frames_per_token)
        self._horizon = int(model.gen_horizon_len)
        self._max_window = (int(10 * self._fps) // self._patch) * self._patch
        wanted = round(self._fps) if history_frames is None else int(history_frames)
        room = self._max_window - self._horizon
        self._history = max(self._patch, min(wanted, room) // self._patch * self._patch)
        self._steps = int(diffusion_steps or model.diffusion.num_base_steps)
        self._cfg = cfg_weight
        self._names = tuple(str(n) for n in self._skeleton.bone_order_names)
        self._embeddings: dict[str, tuple[Any, Any]] = {}
        self._sessions: dict[str, _Session] = {}

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> ArdyEngine:
        history = env.get("ARDY_HISTORY_FRAMES")
        steps = env.get("ARDY_DIFFUSION_STEPS")
        return cls(
            model_name=env.get("ARDY_MODEL", "soma"),
            device=env.get("ARDY_DEVICE", "cuda"),
            acceleration=env.get("ARDY_ACCELERATION", "none"),
            history_frames=int(history) if history else None,
            diffusion_steps=int(steps) if steps else None,
        )

    def info(self) -> EngineInfo:
        return EngineInfo(
            model=self._model_name,
            fps=self._fps,
            horizon_frames=self._horizon,
            joint_names=self._names,
        )

    def open(self, session_id: str, spec: SessionSpec) -> None:
        for key in spec.keyframes:
            if len(key.joints) != len(self._names):
                raise ValueError(f"a keyframe has {len(key.joints)} joints")
        self._sessions[session_id] = _Session(spec=spec)

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def step(self, session_id: str, keep_until_frame: int) -> list[Frame]:
        torch = self._torch
        session = self._sessions[session_id]
        if keep_until_frame < 0 or session.motion is None:
            session.motion = None
            session.joints = None
            session.first = 0
            history = None
            history_end = -1
            kept = 0
        else:
            last = session.first + int(session.motion.shape[1]) - 1
            history_end = min(last, keep_until_frame)
            # At least one token of history.
            history_end = max(history_end, session.first + self._patch - 1)
            kept = min(history_end - session.first + 1, self._history)
            kept = kept // self._patch * self._patch
            begin = history_end - kept + 1 - session.first
            history = session.motion[:, begin : begin + kept]
        history_start = history_end - kept + 1
        num_frames, mask, observed = self._conditions(
            session.spec, history_start, history_end, kept
        )
        from ardy.tools import seed_everything

        seed_everything(_mix(session.spec.seed, history_end + 1))
        text_feat, text_mask = self._embed(session.spec.text_at(history_end + 1))
        init_translation = None
        init_heading = None
        if history is None:
            x, _, z = contract_to_ardy(session.spec.root[0], session.spec.root[1], 0.0)
            init_translation = torch.tensor([[x, 0.0, z]], dtype=torch.float32, device=self._device)
            init_heading = torch.tensor(
                [session.spec.heading], dtype=torch.float32, device=self._device
            )
        with torch.no_grad():
            samples = self._model.autoregressive_step(
                num_frames=num_frames,
                num_denoising_steps=self._steps,
                motion_mask=mask,
                observed_motion=observed,
                cfg_weight=self._cfg,
                texts=None,
                text_feat=text_feat,
                text_pad_mask=text_mask,
                init_history_sequence=history,
                init_global_translation=init_translation,
                init_first_heading_angle=init_heading,
            )
            decoded = self._rep.inverse(self._rep.unnormalize(samples), is_normalized=False)
        fresh = samples[:, kept:]
        posed = decoded["posed_joints"][:, kept:]
        if session.motion is None:
            session.motion, session.joints, session.first = fresh, posed, 0
        else:
            cut = history_end - session.first + 1
            session.motion = torch.cat([session.motion[:, :cut], fresh], dim=1)
            session.joints = torch.cat([session.joints[:, :cut], posed], dim=1)
        self._trim(session)
        return self._frames(posed, history_end + 1)

    # ------------------------------------------------------------------

    def _conditions(
        self, spec: SessionSpec, history_start: int, history_end: int, kept: int
    ) -> tuple[int, Any, Any]:
        """The window's length and its constraint mask and values, from the
        waypoints and keyframes ahead of the history."""
        torch = self._torch
        from ardy.constraints import (
            FullBodyConstraintSet,
            Root2DConstraintSet,
        )

        ahead = [w.frame for w in spec.waypoints if w.frame > history_end]
        ahead += [k.frame for k in spec.keyframes if k.frame > history_end]
        num_frames = kept + self._horizon
        if ahead:
            future = self._max_window - kept - self._horizon
            num_frames = max(num_frames, max(ahead) - history_start + 1)
            num_frames = min(num_frames, future + kept + self._horizon)
            num_frames = math.ceil(num_frames / self._patch) * self._patch
        num_frames = max(min(num_frames, self._max_window), kept + self._horizon)
        end = history_start + num_frames
        constraints: list[Any] = []
        for with_heading in (False, True):
            points = [
                w
                for w in spec.waypoints
                if history_end < w.frame < end and (w.heading is not None) == with_heading
            ]
            if not points:
                continue
            root = []
            for w in points:
                x, _, z = contract_to_ardy(w.x, w.y, 0.0)
                root.append([x, z])
            constraints.append(
                Root2DConstraintSet(
                    self._skeleton,
                    torch.tensor([w.frame for w in points]),
                    torch.tensor(root, dtype=torch.float32, device=self._device),
                    global_root_heading=(
                        torch.tensor(
                            [w.heading for w in points],
                            dtype=torch.float32,
                            device=self._device,
                        )
                        if with_heading
                        else None
                    ),
                )
            )
        keys = [k for k in spec.keyframes if history_end < k.frame < end]
        if keys:
            positions = torch.tensor(
                [[contract_to_ardy(*joint) for joint in k.joints] for k in keys],
                dtype=torch.float32,
                device=self._device,
            )
            rotations = torch.eye(3, device=self._device).expand(len(keys), len(self._names), 3, 3)
            constraints.append(
                FullBodyConstraintSet(
                    self._skeleton,
                    torch.tensor([k.frame for k in keys]),
                    positions,
                    rotations,
                )
            )
        if not constraints:
            return num_frames, None, None
        observed, mask = self._rep.create_conditions_from_constraints(
            constraints, length=end, to_normalize=False, device=self._device
        )
        observed = self._rep.normalize(observed) * mask
        observed = observed[history_start:].clone()
        mask = mask[history_start:].clone()
        observed[:kept] = 0.0
        mask[:kept] = 0.0
        return num_frames, mask.unsqueeze(0), observed.unsqueeze(0)

    def _embed(self, text: str) -> tuple[Any, Any]:
        cached = self._embeddings.get(text)
        if cached is None:
            with self._torch.no_grad():
                cached = self._model._encode_text([text])
            self._embeddings[text] = cached
        return cached

    def _frames(self, posed: Any, first: int) -> list[Frame]:
        from ardy.motion_rep.tools import compute_heading_angle

        headings = compute_heading_angle(posed, self._skeleton)[0].tolist()
        joints = posed[0].tolist()
        root = self._skeleton.root_idx
        frames = []
        for i, (heading, body) in enumerate(zip(headings, joints, strict=False)):
            x, _, z = body[root]
            frames.append(
                Frame(
                    frame=first + i,
                    root=ardy_to_contract(x, 0.0, z),
                    heading=float(heading),
                    joints=tuple(ardy_to_contract(*p) for p in body),
                )
            )
        return frames

    def _trim(self, session: _Session) -> None:
        """Keep no more than two windows of a long session's motion."""
        excess = int(session.motion.shape[1]) - 2 * self._max_window
        if excess > 0:
            session.motion = session.motion[:, excess:]
            session.joints = session.joints[:, excess:]
            session.first += excess


def _mix(seed: int, frame: int) -> int:
    """A step's seed: the session's, and the frame the step starts from."""
    return (seed * 1_000_003 + frame) % (2**31 - 1)
