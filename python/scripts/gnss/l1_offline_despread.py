#!/usr/bin/env python3
"""Independent offline despread of the raw L1 airspy capture -- isolate receiver/clock vs the
kotekan tracker as the source of the per-record carrier-phase decoherence. Absolute-anchored
carrier + code (mirrors the kotekan replica), so if THIS despread coheres record-to-record and
kotekan's doesn't, the bug is in the tracker; if THIS is also random, it's the receiver/clock."""
import numpy as np

FILE = "/Users/kvand/Work/kotekan/airspy_gps/captures/l1_5msps_2026-07-01.bin"
FS = 5.0e6        # sample rate (real)
FOFF = 1.25e6     # carrier IF after the (-1)^n flip = Fs/4
FCARR = 1575.42e6
CHIP = 1.023e6    # C/A chip rate
NMS = 5000        # samples per 1 ms C/A period at 5 MSPS
SECONDS = 10.0

# --- C/A code (IS-GPS-200 G1/G2 LFSR) ---------------------------------------------
G2TAP = {1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),9:(3,10),10:(2,3),
         11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),18:(2,5),19:(3,6),20:(4,7),
         21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),
         31:(3,8),32:(4,9)}
def ca_code(prn):
    g1=[1]*10; g2=[1]*10; t1,t2=G2TAP[prn]; out=np.empty(1023,np.int8)
    for i in range(1023):
        out[i]=g1[9]^g2[t1-1]^g2[t2-1]
        fb1=g1[2]^g1[9]; g1=[fb1]+g1[:9]
        fb2=g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]; g2=[fb2]+g2[:9]
    return (1-2*out).astype(np.float64)  # +-1

def code_samples(prn, n, cp0_chips=0.0, code_rate=CHIP):
    """C/A code sampled over absolute sample indices n (array), starting code phase cp0 (chips),
    chipping at code_rate chips/s (code Doppler folded in). Returns +-1 per sample."""
    c = ca_code(prn)
    ph = (cp0_chips + code_rate * (n / FS)) % 1023.0
    return c[ph.astype(np.int64)]

def main():
    N = int(SECONDS * FS)
    off = 2*NMS  # skip the airspy gain-settle transient (first ~2 ms)
    raw = np.memmap(FILE, dtype=np.int16, mode="r")[off:off+N].astype(np.float64)
    n = np.arange(off, off+N)
    x = raw * (1.0 - 2.0 * (n & 1))          # (-1)^n nyquist flip -> carrier at Fs/4 (ABSOLUTE n)

    # --- robust acquisition: 20-block INCOHERENT accumulation, fine-ish Doppler -----
    print("acquiring (20-block incoherent)...")
    NB=20; nn=np.arange(NMS)
    found=[]
    for prn in range(1,33):
        c=ca_code(prn); Crep=np.conj(np.fft.fft(c[(nn*CHIP/FS%1023).astype(np.int64)]))
        best=0;bd=0
        for dop in np.arange(-5000,5001,200):
            acc=np.zeros(NMS)
            for b in range(NB):
                s=off+b*NMS; seg=x[b*NMS:(b+1)*NMS]
                mix=seg*np.exp(-2j*np.pi*(FOFF+dop)*np.arange(s,s+NMS)/FS)
                acc+=np.abs(np.fft.ifft(np.fft.fft(mix)*Crep))**2
            if acc.max()>best: best=acc.max();bd=dop;sf=acc
        found.append((best/np.median(sf),prn,bd))
    found.sort(reverse=True)
    print("  top PRNs (peak/median, Doppler):")
    for snr,prn,dop in found[:6]:
        print("    PRN %2d  %.1f  dop %+d"%(prn,snr,dop))
    _,prn,dop=found[0]; print("target PRN %d, coarse dop %+d"%(prn,dop))

    # --- refine Doppler (FINE, over 100 ms coherent) + code phase -------------------
    c=ca_code(prn); Crep=np.conj(np.fft.fft(c[(nn*CHIP/FS%1023).astype(np.int64)]))
    # code phase from one block at the coarse Doppler
    s=off; mix=x[:NMS]*np.exp(-2j*np.pi*(FOFF+dop)*np.arange(s,s+NMS)/FS)
    cpidx=int(np.argmax(np.abs(np.fft.ifft(np.fft.fft(mix)*Crep))))
    cp0=(-cpidx*CHIP/FS)%1023.0
    code_rate=CHIP*(1.0+dop/FCARR)
    # fine Doppler: maximise |coherent sum of per-ms A| over a 200 ms window
    def coh200(df):
        tot=0j
        for r in range(200):
            ss=off+r*NMS; idx=np.arange(ss,ss+NMS)
            tot+=np.sum(x[r*NMS:(r+1)*NMS]*np.exp(-2j*np.pi*(FOFF+df)*idx/FS)*code_samples(prn,idx,cp0,code_rate))
        return abs(tot)
    dopf=max(np.arange(dop-200,dop+201,10),key=coh200)
    print("refined: dop %+.0f Hz, code phase %.1f chips"%(dopf,cp0))

    # --- ABSOLUTE-ANCHORED per-1ms despread -> complex A per record -----------------
    nrec = N//NMS
    A=np.empty(nrec,np.complex128)
    for r in range(nrec):
        s=r*NMS; idx=np.arange(s,s+NMS)
        carr=np.exp(-2j*np.pi*(FOFF+dopf)*idx/FS)          # absolute-anchored carrier
        cod=code_samples(prn,idx,cp0,code_rate)             # absolute-anchored code
        A[r]=np.sum(x[idx]*carr*cod)/NMS
    A/=np.abs(A).mean()   # normalise

    # --- coherence of MY despread (the isolation test) -----------------------------
    A2=A*A
    def Rw(W):
        return np.median([np.abs(np.mean(np.exp(1j*np.angle(A2[s:s+W])))) for s in range(0,len(A)-W,W)])
    Rwhole=np.abs(np.mean(np.exp(1j*np.angle(A2))))
    print("\n=== MY independent despread: PRN %d, %d records (1 ms) ==="%(prn,nrec))
    print("  arg(A^2) R: whole=%.3f  200ms(w=200)=%.3f  1s(w=1000)=%.3f  (bias 1/sqrtW: %.3f / %.3f)"%(
        Rwhole,Rw(200),Rw(1000),1/np.sqrt(200),1/np.sqrt(1000)))
    # coherent vs incoherent integration
    for K in (1,10,50,200,1000):
        nb=nrec//K; blk=A[:nb*K].reshape(nb,K)
        coh=np.abs(blk.mean(1)).mean(); inc=np.sqrt((np.abs(blk)**2).mean(1)).mean()
        print("  K=%4d (%4d ms): coherent|<A>|=%.3f  incoherent=%.3f  c/i=%.2f"%(K,K,coh,inc,coh/inc))
    print("\n>> c/i staying ~1 as K grows = COHERENT (=> kotekan tracker bug). c/i ~1/sqrt(K) = my")
    print("   despread is ALSO random (=> receiver/clock). R>>bias over 200ms-1s = real coherence.")

if __name__=="__main__":
    main()
