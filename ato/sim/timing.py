"""Exact tick arithmetic for the fixed-step battle loop.

Anything that accumulates a per-tick amount until it crosses a threshold — DP,
SP, attack cooldowns — must not do it in floating point. At 30 ticks per second
``sum(30 x 1/30) == 0.99999999999999989``, so a one-second period fires on tick
31 instead of 30 and every such timer in the simulator runs about 3% slow.

That error is systematic rather than noisy, which is what makes it dangerous:
calibration does not reveal it, it *absorbs* it. A search over the calibration
constants will happily buy back the missing DP by making enemies slower, and
the simulator then looks accurate on the traces it was fitted to while every
constant in it is wrong.

Two rules keep it exact, and both are cheaper than the float version:

* A **period** becomes an exact integer ratio once, and the accumulator that
  counts toward it is an ``int``. See :class:`PeriodicGrant`.
* A **rate** is accumulated in *tick units*: add the per-second rate once per
  tick and divide by the tick rate only when comparing against a threshold.
  Summing ``1.0`` thirty times is exact; summing ``1/30`` thirty times is not.
  SP is stored this way (:attr:`ato.sim.entities.OperatorUnit.sp_ticks`).
"""

from __future__ import annotations

import math
from fractions import Fraction

#: Denominators above this are treated as float noise rather than data. Level
#: files hold decimal literals with at most a few places, so a value whose exact
#: ratio needs a bigger denominator than this is a binary-expansion artefact.
_MAX_DENOMINATOR = 1_000_000


def exact_ratio(value: float) -> tuple[int, int]:
    """``0.7 -> (7, 10)``, ``1.0 -> (1, 1)``, ``999999.0 -> (999999, 1)``.

    Goes through the *decimal* spelling (``repr``) rather than the binary
    expansion: the game data holds decimal literals, so ``0.7`` means seven
    tenths and not ``0.69999999999999995559...``.
    """
    if not math.isfinite(value):
        return 0, 1
    frac = Fraction(repr(float(value))).limit_denominator(_MAX_DENOMINATOR)
    return frac.numerator, frac.denominator


class PeriodicGrant:
    """Grants one whole unit every ``period`` seconds, in integer arithmetic.

    ``rate_scale`` multiplies the *rate*, not the period: the Adverse rune that
    doubles DP recovery is ``rate_scale=2.0``, which halves the wait.

    A period shorter than one tick is handled by granting several units in the
    same tick, so the accumulator is correct even at extreme scales.
    """

    __slots__ = ("_step", "_threshold", "_acc")

    def __init__(self, period: float, ticks_per_second: int, rate_scale: float = 1.0) -> None:
        pn, pd = exact_ratio(period)
        sn, sd = exact_ratio(rate_scale)
        # units/tick = (sn/sd) / ((pn/pd) * tps) -> add sn*pd per tick, grant at sd*pn*tps.
        self._step = sn * pd if pn > 0 and sn > 0 else 0
        self._threshold = sd * pn * ticks_per_second
        self._acc = 0

    @property
    def active(self) -> bool:
        """False when the period or the rate is non-positive: nothing ever accrues."""
        return self._step > 0 and self._threshold > 0

    @property
    def progress(self) -> float:
        """Fraction of the way to the next unit, for observation encoding."""
        return self._acc / self._threshold if self._threshold > 0 else 0.0

    def tick(self) -> int:
        """Advance one tick. Returns how many whole units came due."""
        if not self.active:
            return 0
        self._acc += self._step
        if self._acc < self._threshold:
            return 0
        n, self._acc = divmod(self._acc, self._threshold)
        return n


def ticks_for(seconds: float, ticks_per_second: int) -> int:
    """Whole ticks that ``seconds`` spans, rounded up.

    The epsilon absorbs the representation error of a value that is *meant* to
    be a whole number of ticks: ``0.1 * 30`` is ``3.0000000000000004``, and a
    bare ``ceil`` would turn a 3-tick delay into 4.
    """
    return max(0, math.ceil(seconds * ticks_per_second - 1e-9))
