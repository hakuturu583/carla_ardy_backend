# The pedestrian motion contract (vendored)

`pedestrian_motion/v0/pedestrian_motion.proto` is the contract between
[autoware_carla_scenario](https://github.com/autowarefoundation/autoware_carla_scenario)'s
`SmartPedestrianControllerAction` (the client) and a pedestrian motion generator such as
this one (the server). It belongs to autoware_carla_scenario. This repository keeps
byte-for-byte copies of it and of the Python modules autoware_carla_scenario generates
from it (`src/carla_ardy_backend/_proto/`). The copies come from the commit named in
`ACS_REF`.

To move to a newer contract:

1. Set `ACS_REF` to the new commit (or release tag).
2. Run `uv run python scripts/check_vendored.py`.
3. Set `CONTRACT_VERSION` in `src/carla_ardy_backend/__init__.py` to the proto header's.

CI runs `scripts/check_vendored.py --check`. While `ACS_REF` is all zeros, before the
contract's autoware_carla_scenario commit is published, the check is skipped with a warning.
