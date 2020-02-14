#include "frameView.hpp"

#include "gpsTime.h"

#include "fmt.hpp"

#include <set>

frameView::frameView(Buffer* buf, int frame_id) :
    buffer(buf),
    id(frame_id),
    _frame(buffer->frames[id]) {}
