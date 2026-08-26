"""S4: the CM/CL sibling chain -- seed a long-code tracker from this chain's solution.

The sibling despreads the SAME satellite on a different code (GPS L2 CM/CL), so it needs no
search of its own: the visible set, the predicted Doppler and the receiver clock have all been
solved here. It consumes and never feeds back, which is why this stage has no outputs into the
rest of the cycle.

⚠️ THE ANCHOR EPOCH IS EVALUATED SEPARATELY, NOT EXTRAPOLATED. Linear extrapolation back to
utc0 is no cure for orbit curvature (tens of ms over hours), so this runs a SECOND model
evaluation at the fixed anchor epoch, cached per ephemeris refresh -- the anchor never moves,
only the ephemeris does.

@author Keith Vanderlinde
"""

from datetime import datetime, timezone

from gnss_broker.sky import C_LIGHT, brdc_predict
from gnss_broker.transport import _now, _get, _post, _log, _log_rl
