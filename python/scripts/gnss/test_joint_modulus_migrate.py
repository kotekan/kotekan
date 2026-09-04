#!/usr/bin/env python3
# GAP 2: a shorter-code chain joining the receiver must ADD rows, not rebuild the state.
# Reproduces the 2026-08-21 14:16 live failure (5 healthy GPS rows wiped the instant
# gal_e5a was fed). Run: python3 test_joint_modulus_migrate.py -- prints PASS/FAIL.
# CONTROL: reverting receiver.py's migrate to the old rebuild makes this print FAIL.
import sys, random
sys.path.insert(0,'/home/kvand/gnss/kotekan/python/scripts/gnss')
from gnss_broker.receiver import Receiver
import inspect
rx = Receiver() if not inspect.signature(Receiver.__init__).parameters.get('cfg') else Receiver(None)
random.seed(5)
L_GPS, L_E5A = 204600.0, 10230.0     # gps_l5 NH vs a shorter-code chain
clk_true = 151.0
G = [("gps_l5", 1+i) for i in range(5)]
E = [("gal_e5a", 20+i) for i in range(9)]
bias = {k: (i-2)*2.0 for i,k in enumerate(G)}
bias.update({k: (i-4)*1.5 for i,k in enumerate(E)})

st = rx.joint_receiver("1176.45MHz", L_GPS)
t = 0.0
for c in range(60):
    t += 2.0
    for k in G: st.update(k, clk_true+bias[k]+random.gauss(0,0.05), 0.3, t)
    st.gauge()
n_before = len(st._idx); clk_before = st.clk; L_before = st.L
obs_before = {k: st.predicted(k) for k in G}

# THE EVENT: a shorter-code chain joins the receiver
st2 = rx.joint_receiver("1176.45MHz", L_E5A)
n_after = len(st2._idx); clk_after = st2.clk

print(f"  before : n={n_before}  clk={clk_before:8.3f}  L={L_before:.0f}")
print(f"  after  : n={n_after}  clk={clk_after:8.3f}  L={st2.L:.0f}   same object: {st2 is st}")
kept = sum(1 for k in G if k in st2._idx)
dobs = max(abs(st2.predicted(k)-obs_before[k]) for k in G) if kept==len(G) else float('nan')
print(f"  GPS rows kept: {kept}/{len(G)}      max change in predicted(clk+b): {dobs:.6f} chips")
# and the new constellation joins ADDITIVELY
for c in range(20):
    t += 2.0
    for k in G+E: st2.update(k, clk_true+bias[k]+random.gauss(0,0.05), 0.3, t)
    st2.gauge()
print(f"  after feeding E: n={len(st2._idx)}  clk={st2.clk:8.3f}  rate={st2.clk_rate:+.4f}")
errs = [abs(st2.predicted(k)-(clk_true+bias[k])) for k in G+E if k in st2._idx]
print(f"  observable err  max {max(errs):.3f} chips over {len(errs)} sats")
ok = (kept==len(G)) and (dobs < 1e-6) and (len(st2._idx)==len(G)+len(E)) and max(errs) < 1.0
print("\n  VERDICT:", "PASS -- rows preserved, observable invariant, both constellations present"
      if ok else "FAIL")
