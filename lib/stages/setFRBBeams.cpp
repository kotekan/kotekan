#include "Config.hpp"          // for Config
#include "FRBBeams.hpp"
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, metadata_is_chord, CHORD_META_MAX_DIM, CHO...
#include "kotekanLogging.hpp"  // for FATAL_ERROR, DEBUG, INFO
#include "restServer.hpp"      // for restServer, connectionInstance
#include "N2Util.hpp"          // for frameID

#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::Stage;
using kotekan::restServer;
using N2::frameID;

class setFRBBeams : public Stage {
public:
    setFRBBeams(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container);
    ~setFRBBeams();
    void main_thread() override;

    void send_beams(connectionInstance& conn) const;

protected:
    std::vector<FRBBeam> build_grid_beams() const;
    std::vector<FRBBeam> build_seth_beams() const;

private:
    Buffer* out_buf;
    const std::string mode;
    const std::vector<FRBBeam> beam_table;
    const uint32_t num_x;
    const uint32_t num_y;
    const double x_min;
    const double x_max;
    const double y_min;
    const double y_max;
    std::vector<FRBBeam> beams;
};

REGISTER_KOTEKAN_STAGE(setFRBBeams);

setFRBBeams::setFRBBeams(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&setFRBBeams::main_thread, this)),
    mode(config.get<std::string>(unique_name, "mode")),
    beam_table(config.get_default<std::vector<FRBBeam>>(unique_name, "beams", {})),
    num_x(config.get_default<uint32_t>(unique_name, "num_x", 0)),
    num_y(config.get_default<uint32_t>(unique_name, "num_y", 0)),
    x_min(config.get_default<double>(unique_name, "x_min", 0.0)),
    x_max(config.get_default<double>(unique_name, "x_max", 0.0)),
    y_min(config.get_default<double>(unique_name, "y_min", 0.0)),
    y_max(config.get_default<double>(unique_name, "y_max", 0.0)) {
        
    // Get Buffer
    out_buf = get_buffer("out_buf");
    out_buf->register_consumer(unique_name);

    // Check mode & assign num_beams
    if (mode == "manual") {
        if (beam_table.size() == 0)
            FATAL_ERROR("manual mode, but `beams` is empty");
        beams = beam_table;
    } else if (mode == "grid") {
        if (num_x == 0 || num_y == 0)
            FATAL_ERROR("grid mode, but num_x ({:d}) or num_y ({:d}) is 0", num_x, num_y);
        beams = build_grid_beams();
    } else if (mode == "seth") {
        beams = build_seth_beams();
    } else {
        FATAL_ERROR("Unknown mode: {:s}", mode);
    }

    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    rest_server.register_get_callback(unique_name + "/beams",
                                      std::bind(&setFRBBeams::send_beams, this, _1));

    out_buf->allocate_ndarray_frame_desc<double, 2>("FRBBeams", {static_cast<ptrdiff_t>(beams.size()), 3}, {"Beam", "B"});
}

setFRBBeams::~setFRBBeams() {
    restServer& rest_server = restServer::instance();
    rest_server.remove_get_callback(unique_name + "/beams");
}

void setFRBBeams::send_beams(connectionInstance& conn) const {
    nlohmann::json reply = {};
    reply.emplace("beams", beams);
    conn.send_json_reply(reply);
}
    
std::vector<FRBBeam> setFRBBeams::build_grid_beams() const {

    std::vector<FRBBeam> grid_beams(num_x * num_y);

    for (uint32_t by = 0; by < num_y; by++) {
        for (uint32_t bx = 0; bx < num_x; bx++) {
            const uint32_t b = bx + num_x * by;

            double x = (x_min * (num_x - bx - 1) + x_max * bx) / (num_x - 1);
            double y = (y_min * (num_y - by - 1) + y_max * by) / (num_y - 1);
            grid_beams.at(b) = {.idx=b, .x_dir_grid=x, .y_dir_grid=y};
        }
    }

    return grid_beams;
}

std::vector<FRBBeam> setFRBBeams::build_seth_beams() const {
    FATAL_ERROR("seth beams not implemented.");
}

void setFRBBeams::main_thread() {

    frameID frame_id(out_buf);

    while (!stop_thread) {
        double *beam_buf = (double *)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (beam_buf == nullptr)
            break;
        
        for (size_t b = 0; b < beams.size(); b++) {
            beam_buf[3*b+0] = static_cast<double>(beams.at(b).idx);
            beam_buf[3*b+1] = beams.at(b).x_dir_grid;
            beam_buf[3*b+2] = beams.at(b).y_dir_grid;
        }

        out_buf->allocate_new_metadata_object(frame_id);
        const std::shared_ptr<chordMetadata> meta = get_chord_metadata(out_buf, frame_id);

        meta->set_from_frame_desc(out_buf->get_ndarray_frame_desc());

        // If this gets made time-dependent, set fpga_seq_num, and fpga_time_downsampling here.

        meta->check_frame_desc(out_buf->get_ndarray_frame_desc());

        out_buf->mark_frame_full(unique_name, frame_id++);
    } 
}
    

