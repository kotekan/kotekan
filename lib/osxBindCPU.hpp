/**
 * @file osxBindCPU.hpp
 * @brief Utilities to mirror Linux CPU affinity functions on Mac systems.
 *  - sched_getaffinity
 *  - pthread_setaffinity_np
 *  - CPU_ZERO
 *  - CPU_SET
 *  - CPU_ISSET
 */

#ifndef OSXBINDCPU_H
#define OSXBINDCPU_H

#include <pthread.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#include "errors.h"

#define SYSCTL_CORE_COUNT   "machdep.cpu.core_count"

typedef struct cpu_set {
  uint32_t    count;
} cpu_set_t;

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Resets the contents of a CPU affinity set.
  * The target @c cpu_set will be reset to system default affinity.
  * @param[in,out]  cs  Affinity set of type @c cpu_set which will be reset.
 **/
static inline void
CPU_ZERO(cpu_set_t *cs) { cs->count = 0; }

/** @brief Adds a core to the target CPU affinity set.
  * The target @c cpu_set will be reset to system default affinity.
  * @param[in]      num Index of the CPU core to be added to the set of preferred cores.
  * @param[in,out]  cs  Affinity set of type @c cpu_set to be modified.
 **/
static inline void
CPU_SET(int num, cpu_set_t *cs) { cs->count |= (1 << num); }

/** @brief Checks whether a given core is flagged as preferred 
  * in the input @c cpu_set.
  * @param[in]      num Index of the CPU core in question.
  * @param[in]      cs  Affinity set of type @c cpu_set to be queried.
 **/
static inline int
CPU_ISSET(int num, cpu_set_t *cs) { return (cs->count & (1 << num)); }

/** @brief Get the CPU affinity of the given @c pthread_t.
  * @param[in]      pid         Index of the CPU core in question.
  * @param[in]      cpu_size    @e Unused.
  * @param[out]     cpu_set     Affinity set of type @c cpu_set which will be filled.
  * @returns        0 if successful, -1 on error.
 **/
int sched_getaffinity(pid_t pid, size_t cpu_size, cpu_set_t *cpu_set);

/** @brief Sets the CPU affinity of the given @c pthread_t.
  * @param[in]      thread      @c pthread_t to be moved onto the given @c cpu_set_t.
  * @param[in]      cpu_size    Number of CPUs in the system / 8. (Slightly mysterious.)
  * @param[in]      cpu_set     Affinity set of type @c cpu_set to be queried.
  * @returns        0.
 **/
int pthread_setaffinity_np(pthread_t thread, size_t cpu_size, cpu_set_t *cpu_set);

#ifdef __cplusplus
}
#endif

#endif
