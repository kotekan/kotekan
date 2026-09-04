#!/usr/bin/env python3
# Gauge-churn bench (2026-08-21): does a membership change corrupt the clock/rate, and
# does the exact re-reference beat the legacy inflate-and-absorb? Run: python3 bench_gauge_churn.py
# Measured at n=4 with churn every 3 cycles -- the live condition that blocked GAP 2.
import sys, math, random, statistics as st
sys.path.insert(0,'/home/kvand/gnss/kotekan/python/scripts/gnss')
from gnss_broker.state_filter import JointReceiverState as J
import inspect; sig=inspect.signature(J.__init__)

def run(rr, nsat=4, cycles=120, churn_every=3, seed=7):
    random.seed(seed)
    kw = dict(rereference=rr)
    if 'code_len' in sig.parameters: kw['code_len']=10230
    s=J(**kw)
    t=0.0; clk_true=151.0; rate_true=0.0
    pool=[("gal_e5a",20+i) for i in range(nsat+2)]
    bias={k:(i-(len(pool)-1)/2)*2.5 for i,k in enumerate(pool)}
    active=list(pool[:nsat])
    clks=[]; rates=[]; obs_err=[]
    for c in range(cycles):
        t+=2.0; clk_true+=rate_true*2.0
        # churn: swap one satellite in/out periodically (the measured live behaviour)
        if churn_every and c%churn_every==0 and c>10:
            out=active.pop(0); s._drop([out]); active.append(pool[(pool.index(out)+1)%len(pool)])
        for k in active:
            s.update(k, clk_true+bias[k]+random.gauss(0,0.05), 0.3, t)
        s.gauge()
        clks.append(float(s.x[0])); rates.append(float(s.x[1]))
        # observable accuracy: clk+b vs truth, the thing that must stay right
        for k in active:
            i=s._idx.get(k)
            if i is not None: obs_err.append(abs((float(s.x[0])+float(s.x[i]))-(clk_true+bias[k])))
    return clks, rates, obs_err

print(f"{'mode':<14} {'clk range':>10} {'clk sd':>8} {'rate range':>11} {'rate sd':>9} {'obs err p95':>12}")
for rr in (False,True):
    c,r,e=run(rr)
    c=c[20:]; r=r[20:]
    e=sorted(e)
    print(f"{'rereference='+str(rr):<14} {max(c)-min(c):10.3f} {st.pstdev(c):8.3f} "
          f"{max(r)-min(r):11.5f} {st.pstdev(r):9.5f} {e[int(.95*len(e))]:12.4f}")
print("\n  clk/rate should be STABLE (small range+sd) and obs err small in BOTH;")
print("  the fix wins if it shrinks clk/rate wander without hurting observable accuracy.")
