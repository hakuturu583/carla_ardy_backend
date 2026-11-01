"""The contract frame and ARDY's.

The pedestrian motion contract (``proto/pedestrian_motion/v0``) places a body
in a right-handed, z-up frame, headings counter-clockwise from ``+x``.  ARDY
is right-handed and y-up, its ground the xz plane, and a heading of 0 faces
``+z`` (the facing of heading h is ``(sin h, 0, cos h)``).

The cyclic swap ``(x, y, z)_contract = (z, x, y)_ardy`` keeps both
right-handed, and takes ARDY's facing ``(sin h, 0, cos h)`` to the contract's
``(cos h, sin h, 0)``: a heading is the same number in both frames.
"""

from __future__ import annotations

__all__ = ["ardy_to_contract", "contract_to_ardy"]


def contract_to_ardy(x: float, y: float, z: float) -> tuple[float, float, float]:
    """A contract point in ARDY's frame."""
    return y, z, x


def ardy_to_contract(x: float, y: float, z: float) -> tuple[float, float, float]:
    """An ARDY point in the contract frame."""
    return z, x, y
