"""The gRPC server of the pedestrian motion contract, over a motion engine.

``carla-ardy-backend`` loads ARDY (:class:`~carla_ardy_backend.ardy_engine.ArdyEngine`)
and only then listens: a client that can connect can generate.  The engine
serves one call at a time -- one model on one GPU.

Environment:

=========================== ==============================================
``PORT``                    The port to listen on (50061).
``ARDY_MODEL``              The checkpoint (``soma``).
``ARDY_DEVICE``             The torch device (``cuda``).
``ARDY_ACCELERATION``       ``none`` or ``compile``.
``ARDY_HISTORY_FRAMES``     Frames of history a step is generated from
                            (one second).
``ARDY_DIFFUSION_STEPS``    Denoising steps (the model's own).
=========================== ==============================================
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from concurrent import futures
from typing import Any

import grpc

from . import CONTRACT_VERSION, __version__
from ._proto import pedestrian_motion_pb2 as pb
from ._proto import pedestrian_motion_pb2_grpc as pb_grpc
from .engine import Keyframe, MotionEngine, SessionSpec, Span, Waypoint

logger = logging.getLogger(__name__)

__all__ = ["PedestrianMotionServicer", "main", "serve"]


class PedestrianMotionServicer(pb_grpc.PedestrianMotionServicer):
    """``pedestrian_motion.v0.PedestrianMotion`` over *engine*."""

    def __init__(self, engine: MotionEngine) -> None:
        self._engine = engine
        self._lock = threading.Lock()

    def Negotiate(self, request: Any, context: Any) -> Any:
        info = self._engine.info()
        if request.contract_version and request.contract_version != CONTRACT_VERSION:
            logger.warning(
                "a client speaks contract %s; this backend speaks %s",
                request.contract_version,
                CONTRACT_VERSION,
            )
        return pb.NegotiateResponse(
            contract_version=CONTRACT_VERSION,
            backend="ardy",
            backend_version=__version__,
            model=info.model,
            fps=info.fps,
            horizon_frames=info.horizon_frames,
            joint_names=info.joint_names,
        )

    def OpenSession(self, request: Any, context: Any) -> Any:
        joints = len(self._engine.info().joint_names)
        keyframes = []
        for key in request.keyframes:
            if len(key.joints) != joints:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"a keyframe at frame {key.frame} has {len(key.joints)} joints, "
                    f"not the {joints} of joint_names",
                )
            keyframes.append(
                Keyframe(
                    frame=int(key.frame),
                    joints=tuple((j.x, j.y, j.z) for j in key.joints),
                )
            )
        spec = SessionSpec(
            root=(request.root.x, request.root.y),
            heading=request.heading,
            seed=int(request.seed),
            texts=tuple(Span(int(t.start_frame), t.text) for t in request.texts),
            waypoints=tuple(
                Waypoint(int(w.frame), w.x, w.y, w.heading if w.has_heading else None)
                for w in request.waypoints
            ),
            keyframes=tuple(keyframes),
        )
        with self._lock:
            self._engine.open(request.session_id, spec)
        logger.info(
            "session %s opened (%d texts, %d waypoints, %d keyframes)",
            request.session_id,
            len(spec.texts),
            len(spec.waypoints),
            len(spec.keyframes),
        )
        return pb.OpenSessionResponse()

    def Step(self, request: Any, context: Any) -> Any:
        try:
            with self._lock:
                frames = self._engine.step(request.session_id, int(request.keep_until_frame))
        except KeyError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"no session {request.session_id!r} is open")
        reply = pb.StepResponse()
        for frame in frames:
            out = reply.frames.add(frame=frame.frame, heading=frame.heading)
            out.root.x, out.root.y, out.root.z = frame.root
            for x, y, z in frame.joints:
                out.joints.add(x=x, y=y, z=z)
        return reply

    def CloseSession(self, request: Any, context: Any) -> Any:
        with self._lock:
            self._engine.close(request.session_id)
        return pb.CloseSessionResponse()


def serve(engine: MotionEngine, port: int, host: str = "0.0.0.0") -> tuple[grpc.Server, int]:
    """Start serving *engine* on *host*:*port*; the server and the port bound
    (*port* 0: one the system picks)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    pb_grpc.add_PedestrianMotionServicer_to_server(PedestrianMotionServicer(engine), server)
    bound = server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server, bound


def main() -> None:
    """``carla-ardy-backend``: load ARDY, then serve it until stopped."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from .ardy_engine import ArdyEngine

    engine = ArdyEngine.from_environment(os.environ)
    port = int(os.environ.get("PORT", "50061"))
    server, port = serve(engine, port)
    info = engine.info()
    logger.info(
        "carla_ardy_backend %s serves %s (%.0f fps, %d joints) on :%d",
        __version__,
        info.model,
        info.fps,
        len(info.joint_names),
        port,
    )
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    stopped.wait()
    server.stop(grace=5).wait()
