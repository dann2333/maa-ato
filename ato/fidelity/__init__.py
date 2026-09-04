"""How much the simulator may be believed, measured rather than assumed.

The simulator is a mental model, not an oracle. A human player also runs one --
"if I drop her there she gets melted by that caster" -- but theirs is corrected
by every battle they actually play, and, crucially, they know when they are
unsure. This package gives ATO the same two properties:

* :mod:`ato.fidelity.trace` records a battle in terms of *screen-observable*
  quantities only, so a simulated battle and a real one are comparable at all.
* :mod:`ato.fidelity.divergence` turns a pair of traces into concrete error
  numbers.
* :mod:`ato.fidelity.ledger` accumulates that evidence per stage and per
  mechanic, and gates what the simulator is allowed to be used for.
* :mod:`ato.fidelity.ensemble` asks a different question -- not "is the
  simulator right" but "does this plan depend on the simulator being right".

The gate is the point. A stage whose simulation has not been checked against
reality contributes no training signal at all. Not down-weighted: none.
"""

from ato.fidelity.divergence import Divergence, compare
from ato.fidelity.ledger import FidelityLedger, FidelityRecord, TrustLevel
from ato.fidelity.trace import BattleTrace, TraceRecorder

__all__ = [
    "BattleTrace",
    "Divergence",
    "FidelityLedger",
    "FidelityRecord",
    "TraceRecorder",
    "TrustLevel",
    "compare",
]
