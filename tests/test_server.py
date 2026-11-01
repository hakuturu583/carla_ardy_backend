"""The server, over a fake engine, through a real gRPC channel."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import grpc
import pytest

from carla_ardy_backend import CONTRACT_VERSION
from carla_ardy_backend._proto import pedestrian_motion_pb2 as pb
from carla_ardy_backend._proto import pedestrian_motion_pb2_grpc as pb_grpc
from carla_ardy_backend.engine import EngineInfo, Frame, SessionSpec
from carla_ardy_backend.frames import ardy_to_contract, contract_to_ardy
from carla_ardy_backend.server import serve

ROOT = Path(__file__).resolve().parents[1]


class _Engine:
    """Holds still where the session opened, one frame per step call."""

    def __init__(self) -> None:
        self.specs: dict[str, SessionSpec] = {}
        self.steps: list[tuple[str, int]] = []

    def info(self) -> EngineInfo:
        return EngineInfo(model="fake", fps=30.0, horizon_frames=2, joint_names=("Hips", "Head"))

    def open(self, session_id: str, spec: SessionSpec) -> None:
        self.specs[session_id] = spec

    def step(self, session_id: str, keep_until_frame: int) -> list[Frame]:
        spec = self.specs[session_id]
        self.steps.append((session_id, keep_until_frame))
        x, y = spec.root
        return [
            Frame(
                frame=keep_until_frame + 1 + i,
                root=(x, y, 0.0),
                heading=spec.heading,
                joints=((x, y, 1.0), (x, y, 1.7)),
            )
            for i in range(2)
        ]

    def close(self, session_id: str) -> None:
        self.specs.pop(session_id, None)


@pytest.fixture
def served() -> Iterator[tuple[_Engine, pb_grpc.PedestrianMotionStub]]:
    engine = _Engine()
    server, port = serve(engine, 0, host="127.0.0.1")
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield engine, pb_grpc.PedestrianMotionStub(channel)
    channel.close()
    server.stop(None)


def test_negotiate_answers_the_contract_and_the_model(
    served: tuple[_Engine, pb_grpc.PedestrianMotionStub],
) -> None:
    _, stub = served
    reply = stub.Negotiate(pb.NegotiateRequest(contract_version=CONTRACT_VERSION))
    assert reply.contract_version == CONTRACT_VERSION
    assert reply.backend == "ardy"
    assert (reply.model, reply.fps, reply.horizon_frames) == ("fake", 30.0, 2)
    assert list(reply.joint_names) == ["Hips", "Head"]


def test_a_session_is_opened_stepped_and_closed(
    served: tuple[_Engine, pb_grpc.PedestrianMotionStub],
) -> None:
    engine, stub = served
    request = pb.OpenSessionRequest(session_id="s", heading=0.5, seed=3)
    request.root.x, request.root.y = 1.0, 2.0
    request.texts.add(start_frame=0, text="walks")
    request.texts.add(start_frame=60, text="waves")
    request.waypoints.add(frame=30, x=3.0, y=4.0)
    request.waypoints.add(frame=90, x=5.0, y=6.0, has_heading=True, heading=1.0)
    key = request.keyframes.add(frame=45)
    key.joints.add(x=1.0, y=2.0, z=1.0)
    key.joints.add(x=1.0, y=2.0, z=1.7)
    stub.OpenSession(request)
    spec = engine.specs["s"]
    assert spec.root == (1.0, 2.0) and spec.heading == 0.5 and spec.seed == 3
    assert spec.text_at(59) == "walks" and spec.text_at(60) == "waves"
    assert [(w.frame, w.heading) for w in spec.waypoints] == [(30, None), (90, 1.0)]
    assert spec.keyframes[0].joints == ((1.0, 2.0, 1.0), (1.0, 2.0, 1.7))

    reply = stub.Step(pb.StepRequest(session_id="s", keep_until_frame=-1))
    assert [f.frame for f in reply.frames] == [0, 1]
    assert [j.z for j in reply.frames[0].joints] == [1.0, 1.7]
    assert engine.steps == [("s", -1)]

    stub.CloseSession(pb.CloseSessionRequest(session_id="s"))
    with pytest.raises(grpc.RpcError) as error:
        stub.Step(pb.StepRequest(session_id="s", keep_until_frame=1))
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


def test_a_keyframe_of_the_wrong_joints_is_refused(
    served: tuple[_Engine, pb_grpc.PedestrianMotionStub],
) -> None:
    _, stub = served
    request = pb.OpenSessionRequest(session_id="s")
    request.keyframes.add(frame=1).joints.add(x=0.0, y=0.0, z=0.0)
    with pytest.raises(grpc.RpcError) as error:
        stub.OpenSession(request)
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_a_text_before_the_first_span_is_none() -> None:
    assert SessionSpec(root=(0, 0), heading=0, texts=()).text_at(5) == ""


@pytest.mark.parametrize("point", [(1.0, 2.0, 3.0), (-4.0, 0.5, 1.7)])
def test_the_frames_swap_both_ways(point: tuple[float, float, float]) -> None:
    assert ardy_to_contract(*contract_to_ardy(*point)) == point


def test_ardys_facing_is_the_contracts_at_the_same_heading() -> None:
    import math

    heading = 0.7
    facing = (math.sin(heading), 0.0, math.cos(heading))  # ARDY's, y up
    x, y, z = ardy_to_contract(*facing)
    assert z == 0.0
    assert math.atan2(y, x) == pytest.approx(heading)


def test_the_contract_version_is_the_vendored_protos() -> None:
    header = (ROOT / "proto/pedestrian_motion/v0/pedestrian_motion.proto").read_text()
    match = re.search(r"Contract version: (\S+)", header)
    assert match is not None and match.group(1) == CONTRACT_VERSION
