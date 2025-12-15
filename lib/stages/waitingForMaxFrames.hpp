#include <atomic>

// To shut down Kotekan cleanly after a certain number of frames have
// been processed -- e.g. have been written to file -- I/O stages use
// this counter to determine whether they should shut down Kotekan.
extern std::atomic<int> waiting_for_max_frames;
