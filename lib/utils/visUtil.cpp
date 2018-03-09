#include "visUtil.hpp"
#include <cstring>


// Initialise the serial from a std::string
input_ctype::input_ctype(uint16_t id, std::string serial) {
    chan_id = id;
    std::memset(correlator_input, 0, 32);
    serial.copy(correlator_input, 32);
}

// Copy the visibility triangle out of the buffer of data, allowing for a
// possible reordering of the inputs
void copy_vis_triangle(
    const int32_t * inputdata, const std::vector<uint32_t>& inputmap,
    size_t block, size_t N, gsl::span<cfloat> output
) {

    size_t pi = 0;
    uint32_t bi;
    uint32_t ii, jj;
    float i_sign;
    bool no_flip;

    if(*std::max_element(inputmap.begin(), inputmap.end()) >= N) {
        throw std::invalid_argument("Input map asks for elements out of range.");
    }

    for(auto i = inputmap.begin(); i != inputmap.end(); i++) {
        for(auto j = i; j != inputmap.end(); j++) {

            // Account for the case when the reordering means we should be
            // indexing into the lower triangle, by flipping into the upper
            // triangle and conjugating.
            no_flip = *i <= *j;
            ii = no_flip ? *i : *j;
            jj = no_flip ? *j : *i;
            i_sign = no_flip ? 1.0 : -1.0;

            bi = prod_index(ii, jj, block, N);

            // IMPORTANT: for some reason the buffers are packed as imaginary
            // *then* real so we need to account for that here.
            output[pi]= {(float)inputdata[2 * bi + 1], i_sign * (float)inputdata[2 * bi]};
            pi++;
        }
    }
}

std::tuple<uint32_t, uint32_t, std::string> parse_reorder_single(json j) {
    if(!j.is_array() || j.size() != 3) {
        throw std::runtime_error("Could not parse json item for input reordering: " + j.dump());
    }

    uint32_t adc_id = j[0].get<int>();
    uint32_t chan_id = j[1].get<int>();
    std::string serial = j[2].get<std::string>();

    return std::make_tuple(adc_id, chan_id, serial);
}

std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> parse_reorder(json& j) {

    uint32_t adc_id, chan_id;
    std::string serial;

    std::vector<uint32_t> adc_ids;
    std::vector<input_ctype> inputmap;

    if(!j.is_array()) {
        throw std::runtime_error("Was expecting list of input orders.");
    }

    for(auto& element : j) {
        std::tie(adc_id, chan_id, serial) = parse_reorder_single(element);

        adc_ids.push_back(adc_id);
        inputmap.emplace_back(chan_id, serial);
    }

    return std::make_tuple(adc_ids, inputmap);

}

std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> default_reorder(size_t num_elements) {

    std::vector<uint32_t> adc_ids;
    std::vector<input_ctype> inputmap;

    for(uint32_t i = 0; i < num_elements; i++) {
        adc_ids.push_back(i);
        inputmap.emplace_back(i, "INVALID");
    }

    return std::make_tuple(adc_ids, inputmap);

}

std::tuple<std::vector<uint32_t>, std::vector<input_ctype>>
parse_reorder_default(Config& config, const std::string base_path) {

    size_t num_elements = config.get_int("/", "num_elements");

    try {
        json reorder_config = config.get_json_array(base_path, "input_reorder");

        return parse_reorder(reorder_config);
    }
    catch(const std::exception& e) {
        return default_reorder(num_elements);
    }
}


size_t _member_alignment(size_t offset, size_t size) {
    return (((size - (offset % size)) % size) + offset);
}

struct_layout struct_alignment(
    std::vector<std::tuple<std::string, size_t, size_t>> members
) {

    std::string name;
    size_t size, num, end = 0, max_size = 0;

    std::map<std::string, std::pair<size_t, size_t>> layout;

    for(auto member : members) {
        std::tie(name, size, num) = member;

        // Uses the end of the *last* member
        size_t start = _member_alignment(end, size);
        end = start + size * num;
        max_size = std::max(max_size, size);

        layout[name] = {start, end};
    }

    layout["_struct"] = {0, _member_alignment(end, max_size)};

    return layout;
}


movingAverage::movingAverage(double length) {
    // Calculate the coefficient for the moving average as a halving of the weight
    alpha = 1.0 - pow(2, -1.0 / length);
}


void movingAverage::add_sample(double value) {

    // Special case for the first sample.
    if(!initialised) {
        current_value = value;
        initialised = true;
    } else {
        current_value = alpha * value + (1 - alpha) * current_value;
    }
}

double movingAverage::average() {
    if(!initialised) {
        return NAN;
    }
    return current_value;
}
