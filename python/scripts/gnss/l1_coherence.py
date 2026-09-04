#!/usr/bin/env python3
"""Clean coherence measurement: one despread function used for BOTH acquisition and per-record,
direct code-phase refinement (no FFT-offset conversion). Answers: does the airspy carrier phase
cohere across ms, or not -- receiver/clock vs our processing."""
import numpy as np, sys
sys.path.insert(0,"/private/tmp/claude-501/-Users-kvand-Work-kotekan-airspy-gps/577fbc9b-5707-48bd-b9be-e298ffbe07ab/scratchpad")
from l1_offline_despread import ca_code, FILE

FS=5e6; FOFF=1.25e6; CHIP=1.023e6; FCARR=1575.42e6; NMS=5000

def load(off,N):
    raw=np.memmap(FILE,dtype=np.int16,mode="r")[off:off+N].astype(np.float64)
    k=np.arange(off,off+N); return raw*(1.0-2.0*(k&1))   # (-1)^n flip, absolute-anchored

def despread(x, s0, code, cp0, d, crate):
    """Coherent despread of block x (starts at absolute sample s0): absolute-anchored carrier at
    FOFF+d, code (len-1023 +-1) at phase cp0 chips (at block start) chipping at crate chips/s."""
    k=np.arange(len(x)); carr=np.exp(-2j*np.pi*(FOFF+d)*(s0+k)/FS)
    ph=(cp0 + crate*(k/FS)) % 1023.0
    return np.sum(x*carr*code[ph.astype(np.int64)])

def main():
    off=20*NMS  # skip the airspy gain-settle transient (first ~20 ms) -> steady state
    x=load(off, int(9*FS))
    # acquire: 20-block INCOHERENT (robust, no coherence assumed), all PRNs, then fine Doppler
    NB=20; kk=np.arange(NMS)
    def acc_of(prn,d):
        c=ca_code(prn); Crep=np.conj(np.fft.fft(c[(kk*CHIP/FS%1023).astype(np.int64)]))
        return sum(np.abs(np.fft.ifft(np.fft.fft(x[b*NMS:(b+1)*NMS]*np.exp(-2j*np.pi*(FOFF+d)*np.arange(off+b*NMS,off+b*NMS+NMS)/FS))*Crep))**2 for b in range(NB))
    best=(0,0,0)
    for prn in range(1,33):
        for d in np.arange(-5000,5001,250):
            a=acc_of(prn,d)
            if a.max()>best[0]: best=(a.max(),prn,d)
    _,prn,d0=best; c=ca_code(prn)
    dopf=max(np.arange(d0-250,d0+251,25),key=lambda d:acc_of(prn,d).max())  # fine dop, incoherent
    crate=CHIP*(1+dopf/FCARR)
    # code phase from a STEADY-STATE FFT peak (validated: cp0 = -peakidx*CHIP/FS)
    Crep=np.conj(np.fft.fft(c[(kk*CHIP/FS%1023).astype(np.int64)]))
    corr=np.abs(np.fft.ifft(np.fft.fft(x[:NMS]*np.exp(-2j*np.pi*(FOFF+dopf)*np.arange(off,off+NMS)/FS))*Crep))
    cp0=(-int(np.argmax(corr))*CHIP/FS)%1023
    print("PRN %d: dop %+.0f Hz, code phase %.2f chips (steady-state)"%(prn,dopf,cp0))

    # per-1 ms complex A (absolute-anchored), signal + wrong-PRN noise
    nrec=len(x)//NMS
    cw=ca_code(1 if prn!=1 else 2)
    Asig=np.array([despread(x[r*NMS:(r+1)*NMS], off+r*NMS, c, (cp0+crate*(r*NMS/FS))%1023, dopf, crate) for r in range(nrec)])
    Anoi=np.array([despread(x[r*NMS:(r+1)*NMS], off+r*NMS, cw,(cp0+crate*(r*NMS/FS))%1023, dopf, crate) for r in range(nrec)])
    print("per-1ms |A|: signal=%.0f  noise=%.0f  -> per-record amplitude SNR=%.2f"%(np.abs(Asig).mean(),np.abs(Anoi).mean(),np.abs(Asig).mean()/np.abs(Anoi).mean()))

    # COHERENCE: coherently sum T consecutive 1ms A's; SNR = mean|sum_sig|/mean|sum_noise|.
    # Grows ~sqrt(T) while carrier-coherent; saturates at the coherence time. Nav bit caps at 20 ms.
    print("\n T(ms)  coh-SNR   sqrtT-ideal   (grows ~sqrtT while coherent; flat = decohered at that T)")
    s1=None
    for T in (1,2,3,5,8,12,16,20):
        nb=nrec//T
        ss=np.abs(Asig[:nb*T].reshape(nb,T).sum(1)).mean(); sn=np.abs(Anoi[:nb*T].reshape(nb,T).sum(1)).mean()
        if s1 is None: s1=ss/sn
        print("  %2d    %6.2f     %6.2f"%(T,ss/sn,s1*np.sqrt(T)))

if __name__=="__main__": main()
