"""ARDY, served over autoware_carla_scenario's pedestrian motion contract.

`ARDY <https://github.com/nv-tlabs/ardy>`_ generates human motion
autoregressively, conditioned online by text, root waypoints and full-body
keyframes.  This package serves it over ``pedestrian_motion.v0`` -- the
contract of autoware_carla_scenario's ``SmartPedestrianControllerAction``,
vendored here (``proto/``, ``_proto/``) -- in the image
``ghcr.io/hakuturu583/carla_ardy_backend``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["CONTRACT_VERSION", "__version__"]

#: The pedestrian motion contract version this backend speaks (the vendored
#: proto's header).
CONTRACT_VERSION = "0.1.0"

try:
    __version__ = version("carla-ardy-backend")
except PackageNotFoundError:  # pragma: no cover - a source checkout
    __version__ = "0.0.0"
