#ifndef KOTEKAN_BACKTRACE_HXX
#define KOTEKAN_BACKTRACE_HXX

#include <string>
#include <vector>

namespace kotekan {

void request_backtraces(std::vector<std::string> signals);

} // namespace kotekan

#endif // #ifndef KOTEKAN_BACKTRACE_HXX
