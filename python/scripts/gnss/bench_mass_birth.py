#!/usr/bin/env python3
# Mass-birth bench (2026-08-21): settle on GPS, then birth a Galileo fleet in ONE cycle --
# the operation that diverged the joint state on 2026-08-10 and 2026-08-18. Measures the
# clock step, the rate contamination, and the final observable accuracy.
import sys, random, statistics as st
sys.path.insert(0,'/home/kvand/gnss/kotekan/python/scripts/gnss')
from gnss_broker.state_filter import JointReceiverState as J
import inspect; sig=inspect.signature(J.__init__)

def run(rr, n_g=4, n_new=11, seed=3):
    """Settle on GPS only, then MASS-BIRTH a Galileo fleet (the 08-10 / 08-18 event)."""
    random.seed(seed)
    kw=dict(rereference=rr)
    if 'code_len' in sig.parameters: kw['code_len']=10230
    s=J(**kw); t=0.0; clk=151.0
    G=[("gps_l5",1+i) for i in range(n_g)]
    E=[("gal_e5a",20+i) for i in range(n_new)]
    bias={k:(i-(n_g-1)/2)*2.5 for i,k in enumerate(G)}
    bias.update({k:(i-(n_new-1)/2)*1.8 for i,k in enumerate(E)})
    for c in range(60):                      # settle on GPS
        t+=2.0
        for k in G: s.update(k, clk+bias[k]+random.gauss(0,0.05), 0.3, t)
        s.gauge()
    pre_clk=float(s.x[0]); pre_rate=float(s.x[1])
    pre_obs=[abs((float(s.x[0])+float(s.x[s._idx[k]]))-(clk+bias[k])) for k in G]
    # MASS BIRTH: all Galileo arrive in one cycle
    t+=2.0
    for k in E: s.update(k, clk+bias[k]+random.gauss(0,0.05), 0.3, t)
    s.gauge()
    post_clk=float(s.x[0]); post_rate=float(s.x[1])
    # settle again and measure the observable for EVERYONE
    for c in range(40):
        t+=2.0
        for k in G+E: s.update(k, clk+bias[k]+random.gauss(0,0.05), 0.3, t)
        s.gauge()
    obs=[abs((float(s.x[0])+float(s.x[s._idx[k]]))-(clk+bias[k])) for k in G+E if k in s._idx]
    return dict(clk_step=post_clk-pre_clk, rate_step=post_rate-pre_rate,
                rate_after=float(s.x[1]), n=len(s._idx),
                obs_p95=sorted(obs)[int(.95*len(obs))] if obs else float('nan'),
                obs_max=max(obs) if obs else float('nan'),
                pre_obs_max=max(pre_obs))

print(f"{'mode':<16} {'clk step':>9} {'rate step':>10} {'rate after':>11} {'n':>3} "
      f"{'obs p95':>8} {'obs max':>8}")
for rr in (False,True):
    r=run(rr)
    print(f"{'rereference='+str(rr):<16} {r['clk_step']:+9.3f} {r['rate_step']:+10.5f} "
          f"{r['rate_after']:+11.5f} {r['n']:3d} {r['obs_p95']:8.3f} {r['obs_max']:8.3f}")
print("\n  MASS BIRTH = the 08-10 and 08-18 divergences. A clean birth keeps the clock step")
print("  small, leaves clk_rate near zero (an offset must NOT be read as a rate), and ends")
print("  with every satellite's observable accurate.")
