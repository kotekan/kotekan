#include "../include/gputils/memcpy_kernels.hpp"

#include "../include/gputils/Array.hpp"
#include "../include/gputils/cuda_utils.hpp"
#include "../include/gputils/test_utils.hpp"
#include "../include/gputils/string_utils.hpp"

#include <iostream>


using namespace std;
using namespace gputils;


static void test_memcpy_kernel(long nbytes)
{
    cout << "test_memcpy_kernel(nbytes=" << nbytes << ")" << endl;

    long nelts = nbytes >> 2;
    long nguard = 128;
    
    assert((nbytes % 128) == 0);
    
    Array<int> hsrc({nelts}, af_rhost | af_random);
    Array<int> hdst({nelts + 2*nguard}, af_rhost | af_random);

    Array<int> gsrc = hsrc.to_gpu();
    Array<int> gdst = hdst.to_gpu();
    
    launch_memcpy_kernel(gdst.data + nguard, gsrc.data, nbytes);
    CUDA_PEEK("launch_memcpy_kernel");

    memcpy(hdst.data + nguard, hsrc.data, nbytes);
    assert_arrays_equal(hdst, gdst.to_host(), "hdst", "gdst", {"i"});
}


static void test_memcpy_kernel_2d(long dpitch, long spitch, long width, long height)
{
    cout << "test_memcpy_kernel_2d(dpitch=" << dpitch << ", spitch=" << spitch
	 << ", width=" << width << ", height=" << height << ")" << endl;
    	
    long nguard = 4*(dpitch+spitch) + 128;
    Array<int> hsrc({height*spitch}, af_rhost | af_random);
    Array<int> hdst({height*dpitch + 2*nguard}, af_rhost | af_random);

    Array<int> gsrc = hsrc.to_gpu();
    Array<int> gdst = hdst.to_gpu();

    launch_memcpy_2d_kernel(gdst.data + nguard, dpitch, gsrc.data, spitch, width, height);
    CUDA_PEEK("launch_memcpy_kernel_2d");

    for (long r = 0; r < height; r++) {
	memcpy(hdst.data + nguard + r * (dpitch >> 2),
	       hsrc.data + r * (spitch >> 2),
	       width);
    }
    
    assert_arrays_equal(hdst, gdst.to_host(), "hdst", "gdst", {"i"});
}


static void test_random_memcpy_kernel(long nb_max)
{
    long nb = 128 * rand_int(nb_max/256, nb_max/128+1);
    test_memcpy_kernel(nb);
}


static void test_random_memcpy_kernel_2d(long nb_max)
{
    long wh_target = rand_int(nb_max/2, nb_max+1);
    long width = 128 * long(exp(rand_uniform() * log(wh_target/128.)));
    long height = wh_target / width;
    
    long maxpitch = long(nb_max/height);
    long pmax = (maxpitch - width) / 128;
    long dpitch = width + 128 * rand_int(0,pmax+1);
    long spitch = width + 128 * rand_int(0,pmax+1);
    
    test_memcpy_kernel_2d(dpitch, spitch, width, height);    
}


int main(int argc, char **argv)
{
    for (long nb = 128; nb < 64*1024; nb += 128)
	test_memcpy_kernel(nb);

    for (long nb_max = 128*1024; nb_max <= 4L * 1024L * 1024L * 1024L; nb_max *= 2) {
	cout << "\nnb_max = " << nb_max << " (" << nbytes_to_str(nb_max) << ")" << endl;
	
	test_random_memcpy_kernel(nb_max);
	for (int i = 0; i < 4; i++)
	    test_random_memcpy_kernel_2d(nb_max);
    }
    
    return 0;
}
