#!/usr/bin/env python3
"""Coherent DEEP integration on real airspy C/A -- the machinery the L1C/L5 pilot deep-wipe needs.
Requires the code held ON the correlation peak: here by PEAK-TRACKING (per-record FFT-argmax ->
unwrap -> linear fit -> interpolate), which follows the TRUE code drift (incl. the clock/IF offset
that the geometric code-Doppler misses -- see l1_code_drift.py). Then:
  per-1ms A -> remove smooth carrier drift (an FLL/PLL's job) -> wipe the 20ms nav bit (the pilot
  secondary-overlay analog) -> coherently integrate PAST the bit. SNR builds ~sqrt(T) while coherent.

Result on captures/l1_5msps_2026-07-01.bin, PRN32: per-1ms |A|/noise ~8.9; RAW coherent SNR
saturates ~28 by 20 ms (carrier coherence + nav bit); DEEP wipe builds 28->40->54->72->85 over
50/100/200/500 ms. Off-peak code tracking (e.g. a physics-only code-Doppler feed-forward) collapses
per-1ms |A|/noise to ~1 and this whole curve to a flat ~1.4 -- which was the real deep-wipe failure.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1_code_drift import ca_code, FILE, FS, FOFF, CHIP, NMS

def run(prn=32, off=20*NMS, span_s=9.0):
    raw=np.memmap(FILE,dtype=np.int16,mode="r")[off:off+int(span_s*FS)].astype(np.float64)
    x=raw*(1-2*((off+np.arange(len(raw)))&1)); nrec=len(x)//NMS
    c=ca_code(prn); Crep=np.conj(np.fft.fft(c[(np.arange(NMS)*CHIP/FS%1023).astype(np.int64)]))
    def acc(d): return sum(np.abs(np.fft.ifft(np.fft.fft(
        x[b*NMS:(b+1)*NMS]*np.exp(-2j*np.pi*(FOFF+d)*np.arange(off+b*NMS,off+b*NMS+NMS)/FS))*Crep))**2
        for b in range(20))
    d0=max(np.arange(-5000,5001,250),key=lambda d:acc(d).max())
    dop=max(np.arange(d0-200,d0+201,20),key=lambda d:acc(d).max())
    # per-record correlation (all lags) at the fixed carrier
    def corr_of(code):
        Cr=np.conj(np.fft.fft(code[(np.arange(NMS)*CHIP/FS%1023).astype(np.int64)]))
        out=np.empty((nrec,NMS),complex)
        for r in range(nrec):
            m=r*NMS+np.arange(NMS)
            out[r]=np.fft.ifft(np.fft.fft(x[m]*np.exp(-2j*np.pi*(FOFF+dop)*(off+m)/FS))*Cr)
        return out
    corr=corr_of(c)
    # PEAK-TRACK the code: smooth (linear) fit to the per-record argmax = the true drift line
    pk=np.argmax(np.abs(corr),1)
    lag=np.polyval(np.polyfit(np.arange(nrec),np.unwrap(pk/NMS*2*np.pi)/(2*np.pi)*NMS,1),np.arange(nrec))
    i0=np.floor(lag).astype(int)%NMS; f=lag-np.floor(lag)
    def at(cc): return cc[np.arange(nrec),i0]*(1-f)+cc[np.arange(nrec),(i0+1)%NMS]*f
    A=at(corr); An=at(corr_of(ca_code(1 if prn!=1 else 2)))     # signal + wrong-PRN noise
    def snr(a,an,lbl):
        row=[]; s1=None
        for T in (1,5,10,20,50,100,200,500):
            nb=len(a)//T
            cs=np.abs(a[:nb*T].reshape(nb,T).sum(1)).mean(); ns=np.abs(an[:nb*T].reshape(nb,T).sum(1)).mean()
            if s1 is None: s1=cs/ns
            row.append("T%3d:%5.1f"%(T,cs/ns))
        print("  %-11s %s"%(lbl," ".join(row)))
    print("PRN %d dop %+.0f, peak-tracked (true code drift); per-1ms |A|/noise=%.2f"%(
        prn,dop,np.abs(A).mean()/np.abs(An).mean()))
    snr(A,An,"RAW")
    # DEEP: remove smooth (quadratic) carrier drift, wipe the 20 ms nav bit, integrate past it
    t=np.arange(nrec)*1e-3
    drift=np.polyval(np.polyfit(t,np.unwrap(np.angle(A**2)),2),t)/2
    Ad=A*np.exp(-1j*drift); And=An*np.exp(-1j*drift); Aw=Ad.copy()
    for b in range((nrec+19)//20):
        sl=slice(b*20,b*20+20); s=Ad[sl].sum(); phi=np.angle((Ad[sl]**2).sum())/2
        Aw[sl]=Ad[sl]*(np.sign(np.real(np.exp(-1j*phi)*s)) or 1.0)
    snr(Aw,And,"DEEP wipe")

if __name__=="__main__":
    run()
