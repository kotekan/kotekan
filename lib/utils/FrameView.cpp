#include "FrameView.hpp"

#include "gpsTime.h"

#include "fmt.hpp"

#include <set>

FrameView::FrameView(Buffer* buf, int frame_id) :
    buffer(buf),
    id(frame_id),
    _frame(buffer->frames[id]) {}
