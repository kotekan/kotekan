// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

#include "gpuBufferHandler.hpp"

#include "json.hpp"

using nlohmann::json;

using kotekan::bufferContainer;
using kotekan::Config;

gpuBufferHandler::gpuBufferHandler(kotekan::Config& config, const std::string& unique_name,
                                   kotekan::bufferContainer& host_buffers,
                                   const std::string buffer_entry, const bool producer) :
    producer(producer),
    unique_name(unique_name) {


    // Get the array of input buffers and record the details in @c in_bufs
    json buffer_list = config.get<std::vector<json>>(unique_name, buffer_entry);
    Buffer* buf = nullptr;
    std::string buffer_name;

    // We should never have a frame size of 0, so set the value if it's 0.
    size_t frame_size = 0;

    for (json buffer : buffer_list) {
        if (buffer.is_string()) {
            buffer_name = buffer.get<std::string>();
            buf = host_buffers.get_buffer(buffer_name);
            assert(buf != nullptr);
        } else {
            throw std::runtime_error(
                fmt::format(fmt("Unknown value in in_bufs: {:s}"), buffer.dump()));
        }
        if (frame_size == 0) {
            frame_size = buf->aligned_frame_size;
        } else if (frame_size != buf->aligned_frame_size) {
            throw std::runtime_error(
                fmt::format("Buffers must all have frames of the same size, buffer {:s} has frame "
                            "size {:d}, which doesn't match the size of other buffers {:d}",
                            buf->buffer_name, buf->aligned_frame_size, frame_size));
        }

        if (producer) {
            register_producer(buf, unique_name.c_str());
        } else {
            register_consumer(buf, unique_name.c_str());
        }
        buffers.push_back(buf);
    }

    precondition_buffer_id = 0;
    execute_buffer_id = 0;
    finalize_buffer_id = 0;
}

NextFrameCollection gpuBufferHandler::get_next_frame_precondition() {
    NextFrameCollection frame_collection;

    frame_collection.buf = buffers[precondition_buffer_id].buf;
    frame_collection.frame_id = buffers[precondition_buffer_id].precondition_id;
    if (producer) {
        frame_collection.frame =
            wait_for_empty_frame(frame_collection.buf, unique_name.c_str(),
                                 buffers[precondition_buffer_id].precondition_id++);
    } else {
        frame_collection.frame =
            wait_for_full_frame(frame_collection.buf, unique_name.c_str(),
                                buffers[precondition_buffer_id].precondition_id++);
    }

    precondition_buffer_id = (precondition_buffer_id + 1) % buffers.size();

    return frame_collection;
}

NextFrameCollection gpuBufferHandler::get_next_frame_execute() {
    NextFrameCollection frame_collection;

    frame_collection.buf = buffers[execute_buffer_id].buf;
    frame_collection.frame_id = buffers[execute_buffer_id].execute_id++;
    frame_collection.frame = frame_collection.buf->frames[frame_collection.frame_id];

    execute_buffer_id = (execute_buffer_id + 1) % buffers.size();

    return frame_collection;
}

NextFrameCollection gpuBufferHandler::get_next_frame_finalize() {
    NextFrameCollection frame_collection;

    frame_collection.buf = buffers[finalize_buffer_id].buf;
    frame_collection.frame_id = buffers[finalize_buffer_id].finalize_id;
    frame_collection.frame = frame_collection.buf->frames[frame_collection.frame_id];

    return frame_collection;
}

void gpuBufferHandler::release_frame_finalize() {
    if (producer) {
        mark_frame_full(buffers[finalize_buffer_id].buf, unique_name.c_str(),
                        buffers[finalize_buffer_id].finalize_id++);
    } else {
        mark_frame_empty(buffers[finalize_buffer_id].buf, unique_name.c_str(),
                         buffers[finalize_buffer_id].finalize_id++);
    }
    finalize_buffer_id = (finalize_buffer_id + 1) % buffers.size();
}