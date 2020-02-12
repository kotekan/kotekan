#include "frameView.hpp"
#include "gpsTime.h"
#include "fmt.hpp"

#include <set>

template<typename T>
gsl::span<T> bind_span(uint8_t* start, std::pair<size_t, size_t> range) {
    T* span_start = (T*)(start + range.first);
    T* span_end = (T*)(start + range.second);

    return gsl::span<T>(span_start, span_end);
}

template<typename T>
T& bind_scalar(uint8_t* start, std::pair<size_t, size_t> range) {
    T* loc = (T*)(start + range.first);

    return *loc;
}

frameView::frameView(Buffer* buf, int frame_id) :
    buffer(buf), id(frame_id), _frame(buffer->frames[id]) {} 
