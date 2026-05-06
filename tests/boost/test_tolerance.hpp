#pragma once

// Shared floating-point tolerances for kotekan boost tests.
//
// For *numerical* comparisons (rounding error from float arithmetic). Not
// intended for physically-motivated tolerances. Kept consistent with the
// Python side in tests/tolerance.py.

#include <limits>

namespace kotekan_test {

// Small multiple of machine epsilon for tight numerical comparisons.
// 4 * eps gives a bit of slack for accumulated rounding without being lax.
constexpr float FP32_TOL = 4.0f * std::numeric_limits<float>::epsilon();
constexpr double FP64_TOL = 4.0 * std::numeric_limits<double>::epsilon();

} // namespace kotekan_test
