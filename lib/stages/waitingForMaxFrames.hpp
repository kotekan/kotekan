#include <atomic>

// To shut down Kotekan cleanly after a certain number of frames have
// been processed -- e.g. have been written to file -- I/O stages use
// this counter to determine whether they should shut down Kotekan.
//
// Not a Stage: this header just exposes a shared counter that writer stages
// decrement until zero to trigger a graceful stop.
extern std::atomic<int> waiting_for_max_frames;
