#include "constDataCheck.hpp"

REGISTER_KOTEKAN_PROCESS(constDataCheck);

constDataCheck::constDataCheck(Config& config,
                        const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&constDataCheck::main_thread, this)) {

    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    ref_real = config.get<std::vector<int32_t>>(unique_name, "real");
    ref_imag = config.get<std::vector<int32_t>>(unique_name, "imag");
}

constDataCheck::~constDataCheck() {
}

void constDataCheck::main_thread() {

    int frame_id = 0;
    uint8_t * frame = NULL;
    int num_errors = 0;

    int framect=0;

    while (!stop_thread) {

        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;

        DEBUG("constDataCheck: Got buffer %s[%d]", buf->buffer_name, frame_id);

        bool error = false;
        num_errors = 0;
        int rfr = ref_real[framect % ref_real.size()];
        int rfi = ref_imag[framect % ref_imag.size()];

        for (uint32_t i = 0; i < buf->frame_size/sizeof(int32_t); i += 2) {

            int32_t imag = *((int32_t *)&(frame[i*sizeof(int32_t)]));
            int32_t real = *((int32_t *)&(frame[(i+1)*sizeof(int32_t)]));

            if (real != rfr || imag != rfi) {
                if (num_errors++ < 1000)
                    ERROR("%s[%d][%d] != %d + %di; actual value: %d + %di",
                        buf->buffer_name, frame_id, i/2,
                        rfr, rfi, real, imag);
                error = true;
            }
        }

        if (!error)
            INFO("The buffer %s[%d] passed all checks; contains all (%d + %di)",
                    buf->buffer_name, frame_id,
                    rfr, rfi);
//                    ref_real, ref_imag);

        mark_frame_empty(buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
        framect++;
    }
}
