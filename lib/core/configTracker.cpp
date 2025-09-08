/**
 * @file
 * @brief Translation unit for ConfigTracker.
 *
 * The ConfigTracker implementation is header-only by design to keep related
 * logic colocated and enable inlining where appropriate. This file provides
 * a dedicated translation unit for the class and documents the intent; it also
 * centralizes common includes used by the tracker.
 */

#include "configTracker.hpp"

#include "Config.hpp"
#include "kotekanLogging.hpp"

#include <iomanip>
#include <mutex>
#include <openssl/md5.h>
#include <sstream>

namespace kotekan {} // namespace kotekan
