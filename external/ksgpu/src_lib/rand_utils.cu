#include "../include/ksgpu/rand_utils.hpp"

using namespace std;

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif

std::mt19937 default_rng(137);


vector<double> random_doubles_with_fixed_sum(int n, double sum)
{
    if (n < 1)
	throw runtime_error("random_doubles_with_fixed_sum(): 'n' argument must be >= 1");
    if (sum <= 0.0)
	throw runtime_error("random_doubles_with_fixed_sum(): 'sum' argument must be > 0");

    vector<double> ret(n);

    for (int i = 0; i < n-1; i++) {
	double t = rand_uniform();
	double p = 1.0 / double(n-1-i);
	ret[i] = sum * (1 - pow(t,p));
	sum -= ret[i];
    }

    ret[n-1] = sum;
    return ret;
}


vector<long> random_integers_with_bounded_product(int n, long bound)
{    
    if (n < 1)
	throw runtime_error("random_integers_with_bounded_product(): 'n' argument must be >= 1");
    if (bound < 1)
	throw runtime_error("random_integers_with_bounded_product(): 'bound' argument must be >= 1");

    double target_log = log(rand_uniform(1.01, bound+0.99));
    vector<double> x = random_doubles_with_fixed_sum(n, target_log);
    
    vector<long> ret(n);    
    for (int i = 0; i < n; i++)
	ret[i] = long(exp(x[i]) + 0.5);
    
    int i0 = rand_int(0, n);

    for (int i1 = 0; i1 < n; i1++) {
	int i = (i0 + i1) % n;
	
	long p = 1;
	for (int j = 0; j < n; j++)
	    if (j != i)
		p *= ret[j];

	ret[i] = long(exp(target_log-log(p)) + 0.5);
	ret[i] = min(ret[i], bound/p);
	ret[i] = max(ret[i], 1L);
    }

    long p = 1;
    for (int j = 0; j < n; j++)
	p *= ret[j];

    xassert(p > 0);
    xassert(p <= bound);
    return ret;
}


} // namespace ksgpu
