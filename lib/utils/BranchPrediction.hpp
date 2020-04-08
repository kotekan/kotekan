/**
 * @file
 * @brief Explicitely instruct the compiler that a branch of execution is more or less likely than
 * any other.
 *
 * Only make use of this if you are certain that your code is actually a bottleneck and that using
 * this improves performance.
 */

#ifndef BRANCHPREDICTION_HPP
#define BRANCHPREDICTION_HPP

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#endif // BRANCHPREDICTION_HPP
