#include "../include/gputils/system_utils.hpp"
#include "../include/gputils/string_utils.hpp"

#include <thread>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>

// Branch predictor hint
#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif

using namespace std;


namespace gputils {
#if 0
};  // pacify emacs c-mode!
#endif


// -------------------------------------------------------------------------------------------------

// FIXME variants of errstr() appear in many source files; define common version somewhere.
// Maybe it's time for <gputils/asserts.hpp> which defines _unlikely(), assert macros, exception factory functions, etc?

inline string errstr(const string &func_name)
{
    stringstream ss;
    ss << func_name << "() failed: " << strerror(errno);
    return ss.str();
}

template<typename T>
inline string errstr(const string &func_name, const T &arg)
{
    stringstream ss;
    ss << func_name << "(" << arg << ") failed: " << strerror(errno);
    return ss.str();
}


// -------------------------------------------------------------------------------------------------


void mkdir_x(const char *path, int mode)
{
    int err = mkdir(path, mode);
    
    if (_unlikely(err < 0))
	throw runtime_error(errstr("mkdir", path));
}


void mkdir_x(const std::string &path, int mode)
{
    mkdir_x(path.c_str(), mode);
}


void mlockall_x(int flags)
{
    // FIXME low-priority things to add:
    //   - test that 'flags' is a subset of (MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT)
    //   - if mlockall() fails, then exception text should pretty-print flags
    
    int err = mlockall(flags);

    if (_unlikely(err < 0))
	throw runtime_error(errstr("mlockall"));
}


void *mmap_x(void *addr, ssize_t length, int prot, int flags, int fd, off_t offset)
{
    assert(length > 0);
    
    void *ret = mmap(addr, length, prot, flags, fd, offset);

    if (_unlikely(ret == MAP_FAILED))
	throw runtime_error(errstr("mmap"));

    assert(ret != nullptr);  // paranoid
    return ret;
}

	     
void munmap_x(void *addr, ssize_t length)
{
    assert(length > 0);
    
    int err = munmap(addr, length);

    if (_unlikely(err < 0))
	throw runtime_error(errstr("munmap"));
}


void usleep_x(ssize_t usec)
{
    // According to usleep() manpage, sleeping for longer than this is an error!
    static constexpr ssize_t max_usleep = 1000000;
	
    assert(usec >= 0);

    while (usec > 0) {
	ssize_t n = std::min(usec, max_usleep);
	usec -= n;
	
	int err = usleep(n);

	if (_unlikely(err < 0))
	    throw runtime_error(errstr("usleep"));
    }
}


// -------------------------------------------------------------------------------------------------
//
// pin_thread_to_vcpus(vcpu_list)
//
// The 'vcpu_list' argument is a list of integer vCPUs, where I'm defining a vCPU
// to be the scheduling unit in pthread_setaffinity_np() or sched_setaffinity().
//
// If hyperthreading is disabled, then there should be one vCPU per core.
// If hyperthreading is enabled, then there should be two vCPUs per core
// (empirically, always with vCPU indices 2*n and 2*n+1?)
//
// I think that the number of vCPUs and their location in the NUMA hierarchy
// always follows the output of 'lscpu -ae', but AFAIK this isn't stated anywhere.
//
// If 'vcpu_list' is an empty vector, then pin_thread_to_vcpus() is a no-op.


void pin_thread_to_vcpus(const vector<int> &vcpu_list)
{
    if (vcpu_list.size() == 0)
	return;

    // I wanted to argument-check 'vcpu_list', by comparing with the number of VCPUs available.
    //
    // FIXME Calling std::thread::hardware_concurrency() doesn't seem quite right, but doing the "right"
    // thing seems nontrivial. According to 'man sched_setaffinity()':
    //
    //     "There are various ways of determining the number of CPUs available on the system, including:
    //      inspecting the contents of /proc/cpuinfo; using sysconf(3) to obtain the values of the
    //      _SC_NPROCESSORS_CONF and _SC_NPROCESSORS_ONLN parameters; and inspecting the list of CPU
    //      directories under /sys/devices/system/cpu/."
    
    int num_vcpus = std::thread::hardware_concurrency();

    cpu_set_t cs;
    CPU_ZERO(&cs);

    for (int vcpu: vcpu_list) {
	if (_unlikely((vcpu < 0) || (vcpu >= num_vcpus))) {
	    stringstream ss;
	    ss << "gputils: pin_thread_to_vcpus: vcpu=" << vcpu
	       << " is out of range (num_vcpus=" << num_vcpus <<  to_string(num_vcpus) + ")";
	    throw runtime_error(ss.str());
	}
	CPU_SET(vcpu, &cs);
    }

    // Note: pthread_self() always succeeds, no need to check its return value.
    int err = pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    
    if (_unlikely(err != 0)) {
	// If pthread_setaffinity_np() fails, then according to its manpage,
	// it returns an error code, rather than setting 'errno'.

	stringstream ss;
	ss << "pthread_setaffinity_np() failed: " << strerror(err);
	throw runtime_error(ss.str());
    }
}


}  // namespace gputils
