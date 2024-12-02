# See n2k/sk_bias/README.txt for a little documentation on how this code is used.

import sys
import pickle
import itertools
import contextlib

import numpy as np
import scipy.signal
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt


def logspace(lo, hi, n):
    assert n >= 2
    assert 0 < lo < hi
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))


def round_up_to_power_of_two(n):
    assert n > 0
    return (1 << int(np.log2(n-0.5) + 1))


def is_perfect_square(n):
    if n < 1:
        return False
    m = int(n**0.5 + 0.5)
    return n == m**2


def savefig(filename):
    print(f'Writing {filename}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


####################################################################################################


class Pdf:
    _edges = np.array([ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 ])
    _rms_min = 0.07
    _rms_max = 1.0e6
    
    def __init__(self, rms):
        """Represents a quantized, clipped Gaussian PDF.
        Take rms=None to get a 'saturating' PDF, which always takes values +/- 7.
        """

        if rms is not None:
            assert self._rms_min <= rms <= self._rms_max
            cdf_edges = scipy.special.erf(self._edges / (2**0.5 * rms))
        else:
            cdf_edges = np.zeros(7)
            
        self.rms = rms
        self.p = np.zeros(8)
        self.p[0] = cdf_edges[0]
        self.p[-1] = 1 - cdf_edges[-1]
        self.p[1:-1] = cdf_edges[1:] - cdf_edges[:-1]

        # Means <x^n>
        ix = np.arange(8, dtype=float)
        x2 = np.sum(self.p * ix**2)
        x4 = np.sum(self.p * ix**4)
        x6 = np.sum(self.p * ix**6)
        x8 = np.sum(self.p * ix**8)
        
        mean_s1 = 2*x2              # <x^2 + y^2>
        mean_s2 = 2*x4 + 2*x2*x2    # <x^4 + 2x^2y^2 + y^4>
        mean_sk = mean_s2 / mean_s1**2 - 1

        # Var(S1) = Var(x^2 + y^2)
        #         = 2 Var(x^2) 
        #
        # Var(S2) = Var(x^4 + 2x^2y^2 + y^4)
        #         = 2 Var(x^4) + 8 Cov(x^4,x^2y^2) + 4 Var(x^2y^2)
        #
        # Cov(S1,S2) = Cov(x^2 + y^2, x^4 + 2x^2y^2 + y^4)
        #            = 2 Cov(x^2,x^4) + 4 Cov(x^2,x^2y^2)
        #
        # Var(SK) = Var(S2/S1^2)
        #         = < [ (dS2)/S1^2 - 2 (dS1) (S2/S1^3) ]^2 >
        #         = Var(S2)/S1^4 - 4 Cov(S1,S2) (S2/S1^5) + 4 Var(S1) (S2^2/S1^6)

        var_s1 = 2*(x4-x2*x2)
        var_s2 = 2*(x8-x4*x4) + 8*(x6-x2*x4)*x2 + 4*(x4**2-x2**4)
        cov_s12 = 2*(x6-x2*x4) + 4*(x4-x2*x2)*x2
        var_sk = var_s2/mean_s1**4 - 4*cov_s12*mean_s2/mean_s1**5 + 4*var_s1*mean_s2**2/mean_s1**6
        
        self.mu = mean_s1
        self.b_large_n = mean_sk - 1
        self.sigma_large_n = np.sqrt(var_sk)   # sqrt(N) * sigma


    @classmethod
    def from_mu(cls, mu):
        if abs(mu-98) < 1.0e-10:
            return cls(rms=None)
        
        assert mu >= 1.0e-6
        assert mu <= 97.999
        
        log_rms_min = np.log(cls._rms_min) + 1.0e-10   # rms=0.07 gives 0 < mu < 1.0e-6
        log_rms_max = np.log(cls._rms_max) - 1.0e-10   # rms=1.0e6 gives mu > 97.999

        ret = scipy.optimize.bisect(
            lambda log_rms: np.log(cls(np.exp(log_rms)).mu) - np.log(mu),
            log_rms_min, log_rms_max)

        return cls(rms = np.exp(ret))
    

    def __str__(self):
        return f'Pdf(rms={self.rms}, mu={self.mu}, b_large_n={self.b_large_n}, sigma_large_n={self.sigma_large_n})'


##################################   compute_bias() and helpers   ##################################


def ipow(x, n):
    """(x ** 1000) is criminally slow in numpy -- this function is a workaround.
    Unit-tested in test_ipow(), which is run via 'python -m sk_bias test'."""

    assert n >= 0

    if n == 0:
        return np.ones_like(x)
    
    c = None
    y = x
    
    while True:
        # Invariant: at top of loop, we want to return (c * y^n)
        # If c is None, then c == 1
        
        if (n & 1):
            if c is None:
                c = np.copy(y)
            else:
                c *= y

        if n == 1:
            return c

        if y is x:
            y = x*x
        else:
            y *= y
        
        n >>= 1

    
def compute_bias(pdf, n, min_s1=None, max_s1=None, bvec=None):
    """Returns b = <sk>-1.
    
    If 'bvec' is specified, it is a length-(98*n+1) array (i.e. same shape as (p1,p2,p3))
    which is indexed as bvec[S1].
        
    Note:  sk = (n+1)/(n-1) * (n*S2/S1^2 - 1) - b(S1)
    """

    assert isinstance(pdf, Pdf)
    
    if n is None:
        assert min_s1 is None
        assert max_s1 is None
        assert bvec is None
        return pdf.b_large_n
        
    if min_s1 is None:
        min_s1 = 1
    if max_s1 is None:
        max_s1 = 98*n
    
    assert n >= 2    
    assert 1 <= min_s1 <= max_s1 <= 98*n

    # Compute length-(98*n+1) vectors p1, p2.
    #
    #     p1[s] = (probability of S1=s)
    #     p2[s] = (probability of S1=s) * (expectation value of S2, given S1=s)
    # Note: the variable names p1,p2 aren't very good!)
    
    p0 = np.zeros(50)
    for i in range(8):
        p0[i**2] = pdf.p[i]

    p1 = np.convolve(p0, p0)
    p2 = p1 * np.arange(99)**2

    nout = 98*n+1
    npad = round_up_to_power_of_two(nout)
    
    q1 = np.fft.rfft(p1, n=npad)
    q2 = np.fft.rfft(p2, n=npad)

    qn1 = ipow(q1, n-1)
    qn2 = n*qn1*q2
    qn1 *= q1
        
    p1 = np.fft.irfft(qn1)[:nout]
    p2 = np.fft.irfft(qn2)[:nout]

    # Compute bias from (p1, p2).
        
    i, j = min_s1, (max_s1+1)
    
    t = np.sum(p1[i:j])
    assert t >= 1.0e-12
        
    mean_s2_s1 = np.sum(p2[i:j] / np.arange(i,j)**2) / t
    mean_sk = (n+1) / (n-1) * (n*mean_s2_s1 - 1)

    if bvec is not None:
        assert bvec.shape == (98*n+1,)
        mean_sk -= np.dot(p1[i:j], bvec[i:j]) / t

    return mean_sk - 1


#######################################   BiasInterpolator  ########################################


def fit_polynomial(xvec, yvec):
    """Helper for BiasInterpolator. Super-stable and gratuitously slow!
    Unit-tested via test_fit_polynomial(), which is run via 'python -m sk_bias test'"""
    
    d = len(xvec)-1
    assert d >= 0
    assert xvec.shape == yvec.shape == (d+1,)
    assert np.all(xvec[:-1] < xvec[1:])

    mat = np.zeros((d+1,d+1))
    mult = np.zeros(d+1)
    coeffs = np.zeros(d+1)
    residual = np.copy(yvec)
    
    for i in range(d+1):
        row = xvec**i if (i > 0) else np.ones(d+1)
        mult[i] = np.dot(row,row)**(-0.5)
        mat[i,:] = row * mult[i]

        coeffs[i] = np.dot(mat[i,:], residual)
        residual -= coeffs[i] * mat[i,:]
        
    coeffs += np.linalg.solve(mat.T, residual)
    coeffs *= mult
    return coeffs


class BiasInterpolator:
    def __init__(self, *, mu_min=None, mu_max=None, bias_nx=128, bias_nmin=64, sigma_nx=64, remove_zero=False):
        if mu_min is None:
            mu_min = 1.0
        if mu_max is None:
            mu_max = 90.0
        
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.bias_nmin = bias_nmin
        self.remove_zero = remove_zero

        # Reminder: x = log(mu)
        dx = (np.log(mu_max) - np.log(mu_min)) / (min(bias_nx,sigma_nx) - 3)
        self.xmin = np.log(mu_min) - dx
        self.xmax = np.log(mu_max) + dx
        
        self.bias_nx = bias_nx
        self.sigma_nx = sigma_nx
        self.bias_xvec = np.linspace(self.xmin, self.xmax, bias_nx)
        self.sigma_xvec = np.linspace(self.xmin, self.xmax, sigma_nx)

        # y = 1/n
        self.bias_ny = 4   # cubic interpolation
        self.yvec = np.linspace(0, 1/bias_nmin, self.bias_ny)
        self.nlist = [ None ] + [ int(1/y + 0.5) for y in self.yvec[1:] ]
        self.yvec[1:] = 1.0 / np.array(self.nlist[1:])

        # print(f'{self.yvec=}')
        # print(f'{self.nlist=}')
                
        # b = bias, s = sigma
        self.bmat = np.zeros((self.bias_nx, self.bias_ny))
        self.bcoeffs = np.zeros((self.bias_nx, self.bias_ny))
        self.scoeffs = np.zeros(self.sigma_nx)

        # Initialize self.bcoeffs
        for i,x in enumerate(self.bias_xvec):
            mu = np.exp(x)
            pdf = Pdf.from_mu(mu)
            for j in range(self.bias_ny):
                n = self.nlist[j]
                self.bmat[i,j] = compute_bias(pdf, n)
            self.bcoeffs[i,:] = fit_polynomial(self.yvec, self.bmat[i,:])

        # Initialize self.scoeffs
        for i,x in enumerate(self.sigma_xvec):
            mu = np.exp(x)
            pdf = Pdf.from_mu(mu)
            self.scoeffs[i] = pdf.sigma_large_n
        
        # Initialize self.binterp, a list of KroghInterpolator objects:
        #   - self.binterp[i] applies for bias_xvec[i+1] <= x <= bias_xvec[i+2]
        #   - self.binterp[i](x) is a length-4 vector containing y-coeffs.

        self.binterp = [ ]
        for i in range(self.bias_nx-3):
            x = scipy.interpolate.KroghInterpolator(self.bias_xvec[i:(i+4)], self.bcoeffs[i:(i+4),:])
            self.binterp.append(x)

        # Initialize self.sinterp, a list of KroghInterpolator objects:
        #   - self.sinterp[i] applies for sigma_xvec[i+1] <= x <= sigma_xvec[i+2]
        #   - self.sinterp[i](x) is a scalar.

        self.sinterp = [ ]
        for i in range(self.sigma_nx-3):
            x = scipy.interpolate.KroghInterpolator(self.sigma_xvec[i:(i+4)], self.scoeffs[i:(i+4)])
            self.sinterp.append(x)
        

    def _setup_x_interpolation(self, mu, nx):
        """Returns x, i."""

        assert (self.mu_min - 1.0e-7) <= mu <= (self.mu_max + 1.0e-7)

        x = np.log(mu)
        t = (nx-1) * (x-self.xmin) / (self.xmax-self.xmin)
        i = int(t)-1
        i = max(i,0)
        i = min(i,nx-4)
        return x, i
    
        
    def interpolate_bias(self, *, mu, n):
        x, i = self._setup_x_interpolation(mu, self.bias_nx)
        assert (self.bias_xvec[i+1] - 1.0e-7) <= x <= (self.bias_xvec[i+2] + 1.0e-7)
        
        y = 0.0 if (n is None) else (1.0/n)
        assert (n is None) or (n >= self.bias_nmin)

        c = self.binterp[i](x)
        assert len(c) == self.bias_ny

        ret = 0
        ypow = 1
        for j in range(self.bias_ny):
            ret += c[j] * ypow
            ypow *= y

        return ret

    
    def interpolate_sigma(self, *, mu):
        x, i = self._setup_x_interpolation(mu, self.sigma_nx)
        assert (self.sigma_xvec[i+1] - 1.0e-7) <= x <= (self.sigma_xvec[i+2] + 1.0e-7)
        return self.sinterp[i](x)


    def get_interpolated_bvec(self, n):
        assert n is not None
        
        eps = 5.0e-11
        min_s1 = n * self.mu_min
        min_s1 = int((1-eps)*min_s1 + 1)  # round up        
        max_s1 = n * self.mu_max
        max_s1 = int((1+eps)*max_s1)      # round down
        assert 0 < min_s1 <= max_s1 <= 98*n
        
        bvec = np.zeros(98*n+1)
        for s1 in range(min_s1, max_s1+1):
            bvec[s1] = self.interpolate_bias(mu=s1/n, n=n)

        return min_s1, max_s1, bvec
    

###########################################   Run MCs   ############################################
#
# python -m sk_bias run_mcs
# python -m sk_bias run_unquantized_mcs
# python -m sk_bias run_transit_mcs


class MCTracker:
    """Helper class for run_mcs(), run_unquantized_mcs(), run_transit_mcs().
    Unit-tested in test_mc_tracker(), which is called via 'python -m sk_bias test'."""

    def __init__(self):        
        self.n = 0
        self.mean = 0
        self.var = 0

    
    def update(self, s):
        assert s.ndim == 1
        
        if len(s) == 0:
            return
        
        ns = len(s)
        smean = np.mean(s)
        
        ds = s - smean
        ssum2 = np.dot(ds,ds)

        no = self.n
        omean = self.mean
        osum2 = (no-1) * self.var

        n = no + ns
        mean = (no*omean + ns*smean) / n
        sum2 = ssum2 + osum2 + no*(omean-mean)**2 + ns*(smean-mean)**2

        self.n = n
        self.mean = mean
        self.var = (sum2/(n-1)) if (n > 1) else 0
        

def run_mcs(pdf, n, *, binterp=True, nbatch=None, mu_min=None, mu_max=None):
    """Called via 'python -m sk_bias run_mcs'. Runs forever!"""

    print(f'run_mcs: called with rms={pdf.rms} {n=} {binterp=} {nbatch=} {mu_min=} {mu_max=}')
    assert binterp in [ True, False ]

    # Initialize min_s1, max_s1, bvec
    if binterp:
        b = BiasInterpolator(mu_min=mu_min, mu_max=mu_max)
        min_s1, max_s1, bvec = b.get_interpolated_bvec(n)
    else:
        min_s1 = max(round(mu_min*n), 1) if (mu_min is not None) else 1
        max_s1 = min(round(mu_max*n), 98*n) if (mu_max is not None) else (98*n)
        bvec = None
        
    if nbatch is None:
        nbatch = 2**21 // n

    print(f'run_mcs: updated mu_min={min_s1/n} mu_max={max_s1/n}')
    assert 1 <= min_s1 <= max_s1 <= 98*n
    
    predicted_bias = compute_bias(pdf, n, min_s1, max_s1, bvec)
    predicted_sigma = pdf.sigma_large_n / n**0.5
    tracker = MCTracker()
        
    print(f'run_mcs: {pdf}')
    print(f'Running Monte Carlos with {n=}')

    for iouter in itertools.count(1):
        if pdf.rms is not None:
            e = np.random.normal(scale=pdf.rms, size=(nbatch,n,2))
            e = np.round(e)
            e = np.maximum(e, -7.)
            e = np.minimum(e, 7.)
        else:
            e = np.full((nbatch,n,2), fill_value=7.0)

        e2 = np.sum(e**2, axis=2)
        s1 = np.sum(e2, axis=1)
        s1 = np.array(s1 + 0.5, dtype=int)   # convert float->int
        s2 = np.sum(e2**2, axis=1)

        valid = np.logical_and(s1 >= min_s1, s1 <= max_s1)
        s1 = s1[valid]
        s2 = s2[valid]
        
        assert np.all(s1 > 0)
        sk = (n+1)/(n-1) * (n*s2/s1**2 - 1)

        if bvec is not None:
            assert bvec.shape == (98*n+1,)
            sk -= bvec[s1]

        tracker.update(sk - 1.0)

        if is_perfect_square(iouter):
            nmc = tracker.n
            pvalid = nmc / (iouter * nbatch)
            delta = tracker.mean - predicted_bias
            rms = np.sqrt(tracker.var) if (tracker.var > 0) else 0.0
            ivar = (1.0 / tracker.var) if (tracker.var > 0) else 0.0
            sigmas = delta * (ivar * nmc)**0.5
            print(f'    nmc={nmc}  {pvalid=}  mean_bias={tracker.mean}'
                  + f'  predicted_mean={predicted_bias}  delta={delta} ({sigmas:.04f} sigma)'
                  + f'  {rms=}  rms_large_n={predicted_sigma}  ratio={rms/predicted_sigma}')


def run_unquantized_mcs(n):
    """Called via 'python -m sk_bias run_unquantized_mcs <n>'. Runs forever!
    Checks some expressions in the notes for <SK> and Var(SK) to all orders in n."""

    nbatch = (2**21 // n) + 1
    tracker = MCTracker()
    print(f'run_unquantized_mcs({n=})')

    for iouter in itertools.count(1):
        e = np.random.normal(size=(nbatch,n,2))
        e2 = np.sum(e**2, axis=2)
        s1 = np.sum(e2, axis=1)
        s2 = np.sum(e2**2, axis=1)
        sk = (n+1)/(n-1) * (n*s2/s1**2 - 1)

        tracker.update(sk)

        if is_perfect_square(iouter):
            print(f'   nmc={tracker.n}  mean_sk={tracker.mean}  var_sk={tracker.var}')


def run_transit_mcs(nt, ndish, brightness):
    """Called via 'python -m sk_bias run_unquantized_mcs <n>'. Runs forever!
    Checks some expressions in the notes for <SK> and Var(SK) during a bright source transit."""

    nbatch = (2**21 // (nt*ndish)) + 1
    tracker = MCTracker()
    print(f'run_transit_mcs({nt=}, {ndish=}, {brightness=})')

    var0 = 4./nt/(2*ndish)
    var1 = var0 * (1 + (ndish-1) * brightness**4 / (1+brightness)**4)
    
    for iouter in itertools.count(1):
        e = np.random.normal(size=(nbatch,2,ndish,nt,2))                       # noise
        e += np.random.normal(size=(nbatch,2,1,nt,2), scale=brightness**0.5)   # source
        e2 = np.sum(e**2, axis=4)     # (mc,pol,dish,t)
        s1 = np.sum(e2, axis=3)       # (mc,pol,dish)
        s2 = np.sum(e2**2, axis=3)    # (mc,pol,dish)
        
        sk = (nt+1)/(nt-1) * (nt*s2/s1**2 - 1)   # (mc,pol,dish)
        sk = np.mean(sk, axis=2)                 # (mc,pol)
        sk = np.mean(sk, axis=1)                 # (mc,)
        assert sk.shape == (nbatch,)
        
        tracker.update(sk)

        if is_perfect_square(iouter):
            print(f'   nmc={tracker.n}  mean_sk={tracker.mean}  var_sk={tracker.var}   {var0=}  {var1=}')


####################################   python -m sk_bias test  #####################################


def test_ipow():
    """This test can be run from the command line with 'python -m sk_bias test'."""
    
    for n in range(1,100):
        nelts = np.random.randint(1000, 2000)
        logr = np.random.uniform(-0.1, 0.1, nelts)
        a = np.exp(logr) * np.exp(1j * np.random.uniform(0,2*np.pi,size=nelts))
        
        aslow = a**n
        afast = ipow(a, n)
        epsilon = np.max(np.abs(aslow-afast) / np.exp(n*logr))
        
        print(f'{n=} {epsilon=}')
        assert np.all(epsilon < 1.0e-12)


def test_fit_polynomial():
    for iouter in range(5):
        for d in range(1, 7):
            maxsep = np.exp(np.random.uniform(-5,5))
            dx = np.random.uniform(0.1*maxsep, maxsep, size=d+1)
            xvec = np.cumsum(dx)
            xvec -= np.mean(xvec)
            xvec += np.random.uniform(-10*maxsep, 10*maxsep)
            yvec = np.random.normal(size=d+1)
            
            coeffs = fit_polynomial(xvec, yvec)
            
            zvec = np.zeros(d+1)
            for i in range(d+1):
                zvec += coeffs[i] * xvec**i

            eps = np.max(np.abs(yvec-zvec))
            print(f'test_fit_polynomial({d=}): {eps=}')


def test_mc_tracker():    
    for iouter in range(10):
        ntot = np.random.randint(5, 11)
        x = np.random.normal(size=ntot)
        
        tracker = MCTracker()
        pos = 0
        
        while pos < ntot:
            end = min(pos + np.random.randint(1,4), ntot)
            tracker.update(x[pos:end])
            pos = end
            
        print(f'MCTracker.test() iteration {iouter}')
        print(f'    {tracker.n=} {len(x)=}')
        print(f'    {tracker.mean=} {np.mean(x)=}')
        print(f'    {tracker.var=} {np.var(x,ddof=1)=}')


#############################  python -m sk_bias check_interpolation   #############################

    
def check_bias_interpolation(binterp):
    """Called by 'python -m sk_bias check_interpolation'."""

    print('Comparing interpolated bias to exact bias, on a grid of (mu,n) values')
    print('Note: this does not test for extra finite-n bias due to scatter between (S1/S0) and mu_true')
        
    mu_vec = logspace(binterp.mu_min, binterp.mu_max, 5 * binterp.bias_nx)
        
    nvec = logspace(binterp.bias_nmin, 10 * binterp.bias_nmin, 10*binterp.bias_ny)[1:-1:2]
    nvec = np.array(nvec+0.5, dtype=int)
    # print(f'{nvec=}')
        
    maxdiff = 0
    argmax = 0.0

    for mu in mu_vec:
        pdf = Pdf.from_mu(mu)
            
        for n in nvec:
            b_exact = compute_bias(pdf, n)
            b_interp = binterp.interpolate_bias(mu=mu, n=n)
            diff = np.abs(b_exact - b_interp)
            if maxdiff <= diff:
                maxdiff = diff
                argmax = (mu,n)
                
    print(f'    maxdiff={maxdiff} at (mu,n)={argmax}')

        
def check_sigma_interpolation(binterp):
    """Called by 'python -m sk_bias check_interpolation'."""

    print('Comparing interpolated sigma(SK) to exact sigma(SK), on a grid of mu values.')
    print('Note: this does not test the approximation that finite-n corrections are negligible!')
    
    mu_vec = logspace(binterp.mu_min, binterp.mu_max, 5 * binterp.sigma_nx)        
    maxdiff = 0
    argmax = 0.0
    
    for mu in mu_vec:
        pdf = Pdf.from_mu(mu)
        s_exact = pdf.sigma_large_n
        s_interp = binterp.interpolate_sigma(mu=mu)
        diff = np.abs(s_interp/s_exact - 1)
        if maxdiff <= diff:
            maxdiff = diff
            argmax = mu
            
    print(f'    maxdiff={maxdiff} at mu={argmax}')
    

####################################################################################################


def make_fig1():
    """Called by 'python -m sk_bias make_plots'.
    Makes the plot in the overleaf, showing (b,sigma) vs mu_true."""
    
    n = 64
    nmu = 100
    
    mu_vec = logspace(1.0, 90.0, nmu)
    b_vec = np.zeros(nmu)
    binf_vec = np.zeros(nmu)
    sigma_vec = np.zeros(nmu)
    max_bdiff = 0.0

    for i,mu in enumerate(mu_vec):
        pdf = Pdf.from_mu(mu)
        b_vec[i] = compute_bias(pdf, n, min_s1=1, max_s1=98*n)
        binf_vec[i] = pdf.b_large_n
        sigma_vec[i] = pdf.sigma_large_n
        max_bdiff = max(abs(b_vec[i]-binf_vec[i]), max_bdiff)

    plt.figure(figsize=(4,3))
    plt.semilogx(mu_vec, b_vec, label=f'${n=}$')
    plt.semilogx(mu_vec, binf_vec, label=r'$n=\infty$')
    plt.xlabel(r'$\mu_{\rm true}$')
    plt.ylabel(r'SK-bias $b$')
    plt.legend(loc='lower left')
    savefig(f'fig1_left_panel.pdf')

    plt.figure(figsize=(4,3))
    plt.semilogx(mu_vec, sigma_vec)
    plt.xlabel(r'$\mu_{\rm true}$')
    plt.ylabel(r'SK variance $\sigma$ ($n=\infty$)')
    savefig(f'fig1_right_panel.pdf')

    print(f'Note: max Delta(b) = {max_bdiff}')


def make_fig2(mu_min, mu_max):
    """Called by 'python -m sk_bias make_plots'.
    Makes the plot in the overleaf, showing (b,sigma) vs mu_obs."""

    # Defaults from __main__ are (mu_min, mu_max) = (2, 50).
    print(f'make_fig2: {mu_min=} {mu_max=}')
    
    nmu = 100
    
    # Pairs (n, color)
    todo = [
        (64, (1, 0, 0)),
        # (85, (0.66, 0, 0.33)),
        # (107, (0.33, 0, 0.66)),
        # (128, (0, 0, 1)),
        (256, (0, 0, 1)),
        # (512, (0, 0.66, 0.33)),
        (1024, (0, 1, 0))
    ]

    nn = len(todo)
    
    mu_vec = logspace(mu_min, mu_max, nmu)
    pdf_list = [ Pdf.from_mu(mu) for mu in mu_vec ]
    sigma_vec = np.array([ pdf.sigma_large_n for pdf in pdf_list ])
    bias_mat = np.zeros((nn,nmu))
    binterp = BiasInterpolator(mu_min=mu_min, mu_max=mu_max)
    
    for (i,(n,_)) in enumerate(todo):
        min_s1, max_s1, bvec = binterp.get_interpolated_bvec(n)
        bias_mat[i,:] = np.array([ compute_bias(pdf, n, min_s1, max_s1, bvec) for pdf in pdf_list ])

    plt.figure(figsize=(4,3))
    for (b,(n,color)) in zip(bias_mat, todo):
        # Plot positive and negative parts
        eps = n**0.5 * np.abs(b) / sigma_vec
        plt.loglog(mu_vec, eps, color=color, ls='-', label=f'n={n}')
    plt.axhline(1.0/2048**0.5, color='k', ls=':', label=r'$2048^{-1/2}$')
    plt.xlabel(r'$\mu_{\rm true}$')
    plt.ylabel(r'$\epsilon$')
    plt.legend(loc='upper left')
    plt.ylim(1.0e-4, 1.0)
    savefig('fig2_right_panel.pdf')
    
    # plt.figure(figsize=(4,3))
    
    
    
#################################   python -m sk_bias emit_code   ##################################


def emit_code():
    hpp_filename = 'sk_globals.hpp'
    cu_filename = 'sk_globals.cu'
    
    interp = BiasInterpolator()
    bias_nx = interp.bias_nx
    bias_ny = interp.bias_ny
    sigma_nx = interp.sigma_nx
    num_debug_checks = 100

    xmin = np.log(interp.mu_min)
    xmax = np.log(interp.mu_max)
    ymax = 1.0 / interp.bias_nmin
    debug_x = np.random.uniform(xmin, xmax, size=num_debug_checks)
    debug_y = np.random.uniform(0.0, ymax, size=num_debug_checks)
    debug_b = [ interp.interpolate_bias(mu=np.exp(x), n=(1.0/y)) for x,y in zip(debug_x,debug_y) ]
    debug_s = [ interp.interpolate_sigma(mu=np.exp(x)) for x in debug_x ]
    
    print(f'Writing {hpp_filename}')
    with open(hpp_filename,'w') as f:
        with contextlib.redirect_stdout(f):
            print('// Autogenerated by python -m sk_bias emit_code')
            print()
            print('#ifndef _N2K_SK_GLOBALS_HPP')
            print('#define _N2K_SK_GLOBALS_HPP')
            print()
            print('namespace n2k {')
            print('namespace sk_globals {')
            print('\n')
            print(f'static constexpr double mu_min = {interp.mu_min};')
            print(f'static constexpr double mu_max = {interp.mu_max};')
            print(f'static constexpr double xmin = {interp.xmin};')
            print(f'static constexpr double xmax = {interp.xmax};')
            print()
            print(f'static constexpr int bias_nx = {interp.bias_nx};')
            print(f'static constexpr int bias_ny = {interp.bias_ny};')
            print(f'static constexpr int bias_nmin = {interp.bias_nmin};')
            print(f'static constexpr int sigma_nx = {interp.sigma_nx};')
            print()
            print('// Returns a pointer to these arrays (back-to-back in memory):')
            print('//   double bias_coeffs[bias_nx][bias_ny];')
            print('//   double sigma_coeffs[sigma_nx];')
            print('extern const double *get_bsigma_coeffs();')
            print()
            print('// Enables a unit test for consistency between python/C++ interpolators')
            print("// The (x,y,b,s) arrays are all 1-d arrays of length 'num_debug_checks'.")
            print(f'static constexpr int {num_debug_checks = };')
            print('extern const double *get_debug_x();')
            print('extern const double *get_debug_y();')
            print('extern const double *get_debug_b();')
            print('extern const double *get_debug_s();')
            print('\n')
            print('}}  // namespace n2k::sk_globals')
            print()
            print('#endif  //  _N2K_SK_GLOBALS_HPP')

    
    print(f'Writing {cu_filename}')
    with open(cu_filename,'w') as f:
        with contextlib.redirect_stdout(f):
            print('// Autogenerated by python -m sk_bias emit_code')
            print()
            print('#include "../include/n2k/internals/sk_globals.hpp"')
            print()
            print('namespace n2k {')
            print('namespace sk_globals {')
            print()
            print(f'// Check that sk_globals.hpp and sk_globals.cu are in sync (paranoid)')
            print(f'static_assert(mu_min == {interp.mu_min});')
            print(f'static_assert(mu_max == {interp.mu_max});')
            print(f'static_assert(xmin == {interp.xmin});')
            print(f'static_assert(xmax == {interp.xmax});')
            print(f'static_assert(bias_nx == {interp.bias_nx});')
            print(f'static_assert(bias_ny == {interp.bias_ny});')
            print(f'static_assert(bias_nmin == {interp.bias_nmin});')
            print(f'static_assert(sigma_nx == {interp.sigma_nx});')
            print(f'static_assert(num_debug_checks == {num_debug_checks});')
            print()
            print(f'static double bsigma_coeffs[{bias_nx}*{bias_ny} + {sigma_nx}] = {{')

            for i in range(bias_nx):
                for j in range(bias_ny):
                    print(f'  {interp.bcoeffs[i,j]:.17g},', end='')
                    if (j == (bias_ny-1)):
                        print()

            for i in range(sigma_nx):
                s = ',' if (i < sigma_nx-1) else ''
                print(f'  {interp.scoeffs[i]:.17g}{s}')
                
            print('};\n')

            for (arr, name) in [ (debug_x,'debug_x'), (debug_y,'debug_y'), (debug_b,'debug_b'), (debug_s,'debug_s') ]:
                print(f'static double {name}[{num_debug_checks}] = {{')
                for i in range(num_debug_checks):
                    s = ',' if (i < num_debug_checks-1) else ''
                    print(f'  {arr[i]:.17g}{s}')
                print('};\n')
                print(f'const double *get_{name}() {{ return {name}; }}\n')

            print('const double *get_bsigma_coeffs() { return bsigma_coeffs; }\n')
            print('}}  // namespace n2k::global_sk')

    print(f"Note: you'll need to copy these autogenerated source files to their proper locations")
    print(f"    cp -f sk_globals.hpp ../include/n2k/internals/sk_globals.hpp")
    print(f"    cp -f sk_globals.cu ../src_lib/sk_globals.cu")
