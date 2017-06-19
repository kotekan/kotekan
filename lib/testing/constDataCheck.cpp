#include "constDataCheck.hpp"

constDataCheck::constDataCheck(Config& config,
                        const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&constDataCheck::main_thread, this)) {

    buf = get_buffer("buf");
    ref_real = config.get_int(unique_name, "real");
    ref_imag = config.get_int(unique_name, "imag");
}

constDataCheck::~constDataCheck() {
}

void constDataCheck::apply_config(uint64_t fpga_seq) {
}

void constDataCheck::main_thread() {

    int buf_id = 0;
    int num_errors = 0;

    for (;;) {

        get_full_buffer_from_list(buf, &buf_id, 1);
        INFO("constDataCheck: Got buffer %s[%d]", buf->buffer_name, buf_id);

        bool error = false;
        num_errors = 0;

        for (uint32_t i = 0; i < buf->buffer_size/sizeof(int32_t); i += 2) {

            int32_t imag = *((int32_t *)&(buf->data[buf_id][i*sizeof(int32_t)]));
            int32_t real = *((int32_t *)&(buf->data[buf_id][(i+1)*sizeof(int32_t)]));

            if (real != ref_real || imag != ref_imag) {
                if (num_errors++ < 10000)
                    ERROR("%s[%d][%d] != %d + %di; actual value: %d + %di\n",
                        buf->buffer_name, buf_id, i/2,
                        ref_real, ref_imag, real, imag);
                error = true;
            }
        }

        if (!error)
            INFO("The buffer %s[%d] passed all checks; contains all (%d + %di)",
                    buf->buffer_name, buf_id,
                    ref_real, ref_imag);

        mark_buffer_empty(buf, buf_id);
        buf_id = (buf_id + 1) % buf->num_buffers;
    }
}