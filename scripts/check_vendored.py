#!/usr/bin/env python
"""Check (or refresh) the vendored pedestrian motion contract.

The contract belongs to autoware_carla_scenario: its ``.proto`` and the
modules generated from it are copied here byte for byte, from the commit
``proto/ACS_REF`` names.  ``--check`` (CI) fails if a copy differs from that
commit's; without it, the copies are refreshed from it.

    uv run python scripts/check_vendored.py --check
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://raw.githubusercontent.com/autowarefoundation/autoware_carla_scenario"

#: Each vendored file, and where autoware_carla_scenario keeps it.
FILES = {
    "proto/pedestrian_motion/v0/pedestrian_motion.proto": (
        "autoware_carla_scenario/proto/pedestrian_motion/v0/pedestrian_motion.proto"
    ),
    **{
        f"src/carla_ardy_backend/_proto/{name}": (
            f"autoware_carla_scenario/src/autoware_carla_scenario/pedestrian/_proto/{name}"
        )
        for name in (
            "pedestrian_motion_pb2.py",
            "pedestrian_motion_pb2.pyi",
            "pedestrian_motion_pb2_grpc.py",
        )
    },
}


def _fetch(ref: str, path: str) -> bytes:
    with urllib.request.urlopen(f"{UPSTREAM}/{ref}/{path}", timeout=30) as response:
        return bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on a difference")
    args = parser.parse_args()
    ref = (ROOT / "proto" / "ACS_REF").read_text().strip()
    if set(ref) == {"0"}:
        # Until the contract's autoware_carla_scenario commit is published,
        # there is nothing to compare with: say so rather than fail.
        print("::warning::proto/ACS_REF is not pinned yet; the vendored contract is unchecked")
        return 0
    stale = []
    for local, upstream in FILES.items():
        want = _fetch(ref, upstream)
        path = ROOT / local
        if path.is_file() and path.read_bytes() == want:
            continue
        stale.append(local)
        if not args.check:
            path.write_bytes(want)
    if not stale:
        print(f"the vendored contract is autoware_carla_scenario@{ref}'s")
        return 0
    if args.check:
        print(f"differs from autoware_carla_scenario@{ref}: {', '.join(stale)}")
        print("refresh with: uv run python scripts/check_vendored.py")
        return 1
    print(f"refreshed from autoware_carla_scenario@{ref}: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
