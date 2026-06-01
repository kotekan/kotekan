#include "Config.hpp"          // for Config
#include "Beams.hpp"           // for FixedBBBeam, TrackingBBBeam
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

constexpr double deg2rad = M_PI / 180.0;

class setBBBeams : public Stage {
public:
    setBBBeams(Config& config, const std::string& unique_name,
                   bufferContainer& buffer_container);
    ~setBBBeams();
    void main_thread() override;

    void send_beams(connectionInstance& conn) const;

protected:
    std::vector<FixedBBBeam> build_grid_beams() const;
    std::vector<FixedBBBeam> build_grid_deg_beams() const;

private:
    Buffer* in_buf;
    Buffer* out_pos_buf;
    Buffer* out_id_buf;
    const std::string fixed_mode;
    const uint32_t num_beams;
    const uint64_t seqs_per_frame;
    const std::vector<FixedBBBeam> fixed_beam_table;
    const std::vector<TrackingBBBeam> tracking_beam_table;
    const uint32_t num_x;
    const uint32_t num_y;
    const double x_min;
    const double x_max;
    const double y_min;
    const double y_max;
    std::vector<FixedBBBeam> fixed_beams;
    std::vector<TrackingBBBeam> tracking_beams;
};

REGISTER_KOTEKAN_STAGE(setBBBeams);

setBBBeams::setBBBeams(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&setBBBeams::main_thread, this)),
    fixed_mode(config.get<std::string>(unique_name, "fixed_mode")),
    num_beams(config.get<uint32_t>(unique_name, "num_beams")),
    seqs_per_frame(config.get<uint64_t>(unique_name, "seqs_per_frame")),
    fixed_beam_table(config.get_default<std::vector<FixedBBBeam>>(unique_name, "fixed_beams", {})),
    tracking_beam_table(config.get_default<std::vector<TrackingBBBeam>>(unique_name, "tracking_beams", {})),
    num_x(config.get_default<uint32_t>(unique_name, "num_x", 0)),
    num_y(config.get_default<uint32_t>(unique_name, "num_y", 0)),
    x_min(config.get_default<double>(unique_name, "x_min", 0.0)),
    x_max(config.get_default<double>(unique_name, "x_max", 0.0)),
    y_min(config.get_default<double>(unique_name, "y_min", 0.0)),
    y_max(config.get_default<double>(unique_name, "y_max", 0.0)) {
        
    // Get Buffer
    in_buf = get_buffer("in_clock_buf");
    in_buf->register_consumer(unique_name);
    out_pos_buf = get_buffer("out_pos_buf");
    out_pos_buf->register_producer(unique_name);
    out_id_buf = get_buffer("out_id_buf");
    out_id_buf->register_producer(unique_name);

    // Check mode & assign num_beams
    if (fixed_mode == "manual") {
        if (fixed_beam_table.size() == 0 && tracking_beam_table.size() == 0)
            FATAL_ERROR("manual mode, but `fixed_beams` and `tracking_beams` are empty");
        fixed_beams = fixed_beam_table;
    } else if (fixed_mode == "grid") {
        if (num_x == 0 || num_y == 0)
            FATAL_ERROR("grid mode, but num_x ({:d}) or num_y ({:d}) is 0", num_x, num_y);
        fixed_beams = build_grid_beams();
    } else if (fixed_mode == "grid_degrees") {
        if (num_x == 0 || num_y == 0)
            FATAL_ERROR("grid_degrees mode, but num_x ({:d}) or num_y ({:d}) is 0", num_x, num_y);
        fixed_beams = build_grid_deg_beams();
    } else {
        FATAL_ERROR("Unknown fixed_mode: {:s}", fixed_mode);
    }
    tracking_beams = tracking_beam_table;

    if (fixed_beams.size() + tracking_beams.size() != num_beams)
        FATAL_ERROR("Number of fixed {:d} + tracking {:d} beams != num_beams {:d}",
                fixed_beams.size(), tracking_beams.size(), num_beams);

    using namespace std::placeholders;
    restServer& rest_server = restServer::instance();
    rest_server.register_get_callback(unique_name + "/beams",
                                      std::bind(&setBBBeams::send_beams, this, _1));

    out_pos_buf->allocate_ndarray_frame_desc<float, 2>("bb_beam_positions", {static_cast<ptrdiff_t>(num_beams), 2}, {"B", "X/Y"});
    out_id_buf->allocate_ndarray_frame_desc<uint64_t, 1>("bb_beam_ids", {static_cast<ptrdiff_t>(num_beams)}, {"R"});
}

setBBBeams::~setBBBeams() {
    restServer& rest_server = restServer::instance();
    rest_server.remove_get_callback(unique_name + "/beams");
}

void setBBBeams::send_beams(connectionInstance& conn) const {
    nlohmann::json reply = {};
    reply.emplace("fixed_beams", fixed_beams);
    reply.emplace("tracking_beams", tracking_beams);
    conn.send_json_reply(reply);
}
    
std::vector<FixedBBBeam> setBBBeams::build_grid_beams() const {

    std::vector<FixedBBBeam> grid_beams(num_x * num_y);

    for (uint32_t by = 0; by < num_y; by++) {
        for (uint32_t bx = 0; bx < num_x; bx++) {
            const uint32_t b = bx + num_x * by;

            double x = (x_min * (num_x - bx - 1) + x_max * bx) / (num_x - 1);
            double y = (y_min * (num_y - by - 1) + y_max * by) / (num_y - 1);
            grid_beams.at(b) = {.id=b, .x_dir_grid=x, .y_dir_grid=y};
        }
    }

    return grid_beams;
}
    
std::vector<FixedBBBeam> setBBBeams::build_grid_deg_beams() const {

    std::vector<FixedBBBeam> grid_beams(num_x * num_y);

    for (uint32_t by = 0; by < num_y; by++) {
        for (uint32_t bx = 0; bx < num_x; bx++) {
            const uint32_t b = bx + num_x * by;

            double x = (x_min * (num_x - bx - 1) + x_max * bx) / (num_x - 1);
            double y = (y_min * (num_y - by - 1) + y_max * by) / (num_y - 1);
            grid_beams.at(b) = {.id=b, .x_dir_grid=sin(x * deg2rad), .y_dir_grid=sin(y * deg2rad)};
        }
    }

    return grid_beams;
}
    

void setBBBeams::main_thread() {

    frameID in_frame_id(in_buf);
    frameID pos_frame_id(out_pos_buf);
    frameID id_frame_id(out_id_buf);

    uint64_t num_frames = 0; // Total number of frame output
    uint64_t seq0 = 0; // seq number of 1st output frame
    bool initialized = false;

    while (!stop_thread) {

        // Grab the input buffer we're using for a clock.
        uint8_t *in_ptr = (uint8_t *)in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (in_ptr == nullptr)
            break;

        // Grab the metadata and unsure it has the fields we need.
        const std::shared_ptr<const chordMetadata> in_meta = get_chord_metadata(in_buf, in_frame_id);
        if (!in_meta->has_fpga_seq_num())
            FATAL_ERROR("in_buf {:s} has no fpga_seq_num, needed for setting clock.",
                    in_buf->buffer_name);

        // Grab the seq num
        uint64_t input_seq = in_meta->get_fpga_seq_num();
        
        // All we need, release the frame.
        in_buf->mark_frame_empty(unique_name, in_frame_id++);

        // First time setup
        if (!initialized) {
            // Set the seq_num of the first output to be on our output cadence and <= the seq_num
            // of the first frame we saw. This means these beams will already be considered valid.
            seq0 = seqs_per_frame * (input_seq / seqs_per_frame);
            initialized = true;
        }

        // Compute the seq_num for this output frame
        uint64_t seq_num = seq0 + num_frames * seqs_per_frame;

        // The input buffer may be much faster than the beam pos & id buffers, so just spin here
        // until we're close to an output time.
        if (seq_num > input_seq)
            continue;

        // TODO: remove this (and update the cuda wrappers) to make the beam positions time dependent.  Have to keep this stage spinning on the input buffer to not stall the pipeline.
        if (num_frames > 0)
            continue;

        // Grab output buffer frames
        float *beam_pos = (float *)out_pos_buf->wait_for_empty_frame(unique_name, pos_frame_id);
        if (beam_pos == nullptr)
            break;
        uint64_t *beam_id = (uint64_t *)out_pos_buf->wait_for_empty_frame(unique_name, id_frame_id);
        if (beam_id == nullptr)
            break;

        // Get the EOP at the center of this frame, needed for tracking beams.
        const Telescope& tel = Telescope::instance();
        const uint64_t t_inst_ns = tel.to_time_ns(seq_num + seqs_per_frame / 2);
        const EOP eop = tel.get_EOP_at_time_ns(t_inst_ns);

        DEBUG("Writing {:d} beams to {:s} and {:s}", fixed_beams.size() + tracking_beams.size(),
                out_pos_buf->buffer_name, out_id_buf->buffer_name);
       
        // Write the fixed beams
        for (size_t b = 0; b < fixed_beams.size(); b++) {
            beam_id[b] = fixed_beams.at(b).id;
            beam_pos[2*b+0] = fixed_beams.at(b).x_dir_grid;
            beam_pos[2*b+1] = fixed_beams.at(b).y_dir_grid;
        }

        // Write the tracking beams
        const size_t b_off = fixed_beams.size();
        for (size_t b = 0; b < tracking_beams.size(); b++) {
            beam_id[b + b_off] = tracking_beams.at(b).id;

            // Get the pointing vector in the grid frame for this beam from its RA and DEC.
            vec3d_t n_grid = tel.vec_cirs_ra_dec_to_grid(tracking_beams.at(b).ra_cirs_deg,
                                                         tracking_beams.at(b).dec_cirs_deg,
                                                         eop);
            // Set the beam position
            beam_pos[2*(b+b_off)+0] = n_grid[0];
            beam_pos[2*(b+b_off)+1] = n_grid[1];
        }

        // Set pos metadata
        out_pos_buf->allocate_new_metadata_object(pos_frame_id);
        const std::shared_ptr<chordMetadata> pos_meta = get_chord_metadata(out_pos_buf, pos_frame_id);

        // Set ndarray sizes and timing
        pos_meta->set_from_frame_desc(out_pos_buf->get_ndarray_frame_desc());
        pos_meta->set_fpga_seq_num(seq_num);
        pos_meta->set_time_downsampling_fpga(seqs_per_frame);

        // Check consistency just in case
        pos_meta->check_frame_desc(out_pos_buf->get_ndarray_frame_desc());

        // Set id metadata
        out_id_buf->allocate_new_metadata_object(id_frame_id);
        const std::shared_ptr<chordMetadata> id_meta = get_chord_metadata(out_id_buf, id_frame_id);

        // Check consistency just in case
        id_meta->set_from_frame_desc(out_id_buf->get_ndarray_frame_desc());
        id_meta->set_fpga_seq_num(seq_num);
        id_meta->set_time_downsampling_fpga(seqs_per_frame);

        // Check consistency just in case
        id_meta->check_frame_desc(out_id_buf->get_ndarray_frame_desc());

        // Increment frames so next loop gets a new seq_num
        num_frames++;

        // Release filled frames.
        out_pos_buf->mark_frame_full(unique_name, pos_frame_id++);
        out_id_buf->mark_frame_full(unique_name, id_frame_id++);
    } 
}
    

