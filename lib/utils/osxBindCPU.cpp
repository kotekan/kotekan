#include "osxBindCPU.hpp"

#include "kotekanLogging.hpp"

int sched_getaffinity(pid_t pid, size_t cpu_size, cpu_set_t* cpu_set) {
    // Unused parameters, FIXME: can we remove them?
    (void)pid;
    (void)cpu_size;

    int32_t core_count = 0;
    size_t len = sizeof(core_count);
    int ret = sysctlbyname(SYSCTL_CORE_COUNT, &core_count, &len, nullptr, 0);
    if (ret) {
        ERROR_NON_OO("error while get core count {:d}\n", ret);
        return -1;
    }
    cpu_set->count = 0;
    for (int i = 0; i < core_count; i++) {
        cpu_set->count |= (1 << i);
    }

    return 0;
}

int pthread_setaffinity_np(pthread_t thread, size_t cpu_size, cpu_set_t* cpu_set) {
    thread_port_t mach_thread;
    size_t core = 0;

    for (core = 0; core < 8 * cpu_size; core++) {
        if (CPU_ISSET(core, cpu_set))
            break;
    }

    if (core > INT_MAX)
        ERROR_NON_OO("Overflow error in osxBindCPU");
    INFO_NON_OO("binding to core {:d}\n", core);
    thread_affinity_policy_data_t policy = {(int)core};
    mach_thread = pthread_mach_thread_np(thread);
    thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1);
    return 0;
}
