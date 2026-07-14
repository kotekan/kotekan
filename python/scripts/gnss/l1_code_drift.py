#!/usr/bin/env python3
"""THE clock-error finding (2026-07-01): on the airspy the code and carrier are on DIFFERENT clocks.

The whole GNSS signal is Doppler-scaled by one fraction v/c at the antenna (code and carrier alike).
What the receiver OBSERVES depends on its clocks:
    carrier_frac (observed carrier Doppler / f_L1)   = v/c - l     (l = tuner-LO clock error)
    code_frac    (observed code drift  / f_chip)      = v/c - a     (a = ADC sample-clock error)
    => code_frac - carrier_frac = l - a               (LO-vs-ADC offset; SAME for every satellite)
Carrier-aiding (code = carrier * f_chip / f_L1) assumes l == a (one shared clock). On the airspy R2
that difference is ~+2.6 ppm, so aiding under-predicts the code drift by ~2.66 chip/s (3x the
geometric code-Doppler). You cannot get (l-a) from the carrier alone (v/c and l are degenerate) --
so the code rate must be tracked EMPIRICALLY (the broker cp_rate slope fit absorbs l-a and its
temperature drift) or calibrated once. An OCXO shrinks (l-a); empirical tracking mops up the residual.

CONFIRMATION for a capture: code_frac - carrier_frac must be the SAME constant for every satellite
(it is the receiver's LO-vs-ADC offset, not per-sat geometry). Run on a multi-strong-sat grab.

Usage:  python3 l1_code_drift.py [capture.bin]   (default: l1_5msps_2026-07-01.bin)
"""
import numpy as np, sys, os

FILE = "/home/lwlab/l1_gx10.bin"
FS = 5.0e6          # assumed sample rate (real)
FOFF = 1.25e6       # IF after the (-1)^n flip = Fs/4
CHIP = 1.023e6      # C/A chip rate
FCARR = 1575.42e6   # L1
NMS = 5000          # samples / 1 ms C/A period at 5 MSPS

# --- C/A code (IS-GPS-200 G1/G2 LFSR) ---
G2TAP = {1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),9:(3,10),10:(2,3),
         11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),18:(2,5),19:(3,6),20:(4,7),
         21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),
         31:(3,8),32:(4,9)}
def ca_code(prn):
    g1=[1]*10; g2=[1]*10; t1,t2=G2TAP[prn]; out=np.empty(1023,np.int8)
    for i in range(1023):
        out[i]=g1[9]^g2[t1-1]^g2[t2-1]
        g1=[g1[2]^g1[9]]+g1[:9]
        g2=[g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]]+g2[:9]
    return (1-2*out).astype(np.float64)

def _load(path, off=20*NMS):
    n_tot=os.path.getsize(path)//2
    raw=np.memmap(path,dtype=np.int16,mode="r")[off:n_tot].astype(np.float64)
    return raw*(1-2*((off+np.arange(len(raw)))&1)), off, (n_tot-off)/FS   # (-1)^n flip, dur

def measure(prn, x, off, dur, nb=250, pm_min=7.0):
    """Robust code-drift + Doppler for one PRN: fit the incoherent code-peak location across many
    epochs spanning the pass (uses only epochs where the sat is up, so rising/setting sats still fit).
    Returns (carrier_dop_Hz, median_pk/med, n_epochs, code_frac_ppm, carrier_frac_ppm) or None."""
    NR=len(x)//NMS
    c=ca_code(prn); Crep=np.conj(np.fft.fft(c[(np.arange(NMS)*CHIP/FS%1023).astype(np.int64)]))
    def acc(d,r0,n): return sum(np.abs(np.fft.ifft(np.fft.fft(
        x[b*NMS:(b+1)*NMS]*np.exp(-2j*np.pi*(FOFF+d)*np.arange(off+b*NMS,off+b*NMS+NMS)/FS))*Crep))**2
        for b in range(r0,r0+n))
    d0=max(np.arange(-5000,5001,200),key=lambda d:acc(d,0,200).max())          # coarse+fine Doppler
    dop=max(np.arange(d0-200,d0+201,15),key=lambda d:acc(d,0,200).max())
    ts=[]; locs=[]; snr=[]
    for e in [int(t*1000) for t in np.arange(1.0, dur-1.2, 3.0)]:              # ~every 3 s
        if e+nb>NR: break
        a=acc(dop,e,nb); k=int(np.argmax(a)); y0,y1,y2=a[(k-1)%NMS],a[k],a[(k+1)%NMS]
        pm=a.max()/np.median(a)
        if pm>pm_min:
            ts.append(e/1000.0); locs.append(k+0.5*(y0-y2)/(y0-2*y1+y2)); snr.append(pm)
    if len(ts)<3: return None
    locu=np.unwrap(np.array(locs)/NMS*2*np.pi)/(2*np.pi)*NMS                   # unwrap wrapping peak
    slope=np.polyfit(np.array(ts),locu,1)[0]                                   # samples/s
    return dop, float(np.median(snr)), len(ts), -slope/(FS/CHIP)/CHIP*1e6, dop/FCARR*1e6

if __name__=="__main__":
    path=sys.argv[1] if len(sys.argv)>1 else FILE
    x,off,dur=_load(path)
    print("capture: %s  (%.1f s)"%(os.path.basename(path),dur))
    print("PRN  carrierDop  pk/med  nEp  code_frac  carr_frac   l-a = code-carr")
    diffs=[]
    for prn in range(1,33):
        try:
            r=measure(prn,x,off,dur)
            if r:
                dop,s,ne,cf,rf=r
                print("%3d   %+6.0f     %4.0f    %2d  %+7.3f    %+7.3f      %+7.3f ppm"%(prn,dop,s,ne,cf,rf,cf-rf))
                diffs.append(cf-rf)
        except Exception:
            pass
    if len(diffs)>=2:
        print("\n(l-a) across %d sats: mean %+.3f ppm, std %.3f ppm -> %s"%(
            len(diffs),np.mean(diffs),np.std(diffs),
            "COMMON => LO-vs-ADC clock offset CONFIRMED" if np.std(diffs)<0.4
            else "NOT common => re-check (per-sat effect / measurement)"))
    elif len(diffs)==1:
        print("\nonly one strong sat (l-a=%+.3f ppm); need >=2 for the common-offset confirmation"%diffs[0])
    else:
        print("\nno sat cleared the SNR floor -- grab from a more open sky / stronger position")
