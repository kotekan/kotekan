"""3c: the fleet delay-lock stage -- poll the discriminator, then run everything hung off it.

This is the shell. It fills `ctx.dllp.fleet` (the cycle's central per-satellite state), calls
the eight `instruments` diagnostics, and finally `codeloop.stage_dll_control`, the only part
that actuates.

⚠️ THE q FLOOR IS MEASURED FROM THIS CYCLE'S OWN POPULATION, NEVER A CONSTANT. Summing across
instances tightens the noise distribution instead of raising q, so the correct bar FALLS as
instances are added: any fixed constant is right for exactly one fleet size.

⚠️ JUDGE LOCK ON q, NEVER ON sig/deep/cn0_coh. Those duty-cycle with the known-rate fold and
will read as lock loss when the tracking is fine.

@author Keith Vanderlinde
"""

import math

from gnss_broker.transport import _now, _log, _log_rl, log_tag
from gnss_broker.fleet import fleet_dll, fleet_coherent
from gnss_broker.fits import q_stall_verdict, instance_stall_verdict
from gnss_broker import instruments
from gnss_broker import codeloop
