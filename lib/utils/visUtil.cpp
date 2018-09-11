#include "visUtil.hpp"
#include <cstring>

// Initialise the serial from a std::string
input_ctype::input_ctype() {
    chan_id = 0;
    std::memset(correlator_input, 0, 32);
}

// Initialise the serial from a std::string
input_ctype::input_ctype(uint16_t id, std::string serial) {
    chan_id = id;
    std::memset(correlator_input, 0, 32);
    serial.copy(correlator_input, 32);
}

bool operator!=(const rstack_ctype& lhs, const rstack_ctype& rhs)
{
    return (lhs.stack != rhs.stack) || (lhs.conjugate != rhs.conjugate);
}

// JSON converters
void to_json(json& j, const freq_ctype& f) {
    j = json{{"centre", f.centre}, {"width", f.width}};
}

void to_json(json& j, const input_ctype& i) {
    j = json{{"chan_id", i.chan_id}, {"correlator_input", i.correlator_input}};
}

void to_json(json& j, const prod_ctype& p) {
    j = json{{"input_a", p.input_a}, {"input_b", p.input_b}};
}

void to_json(json& j, const time_ctype& t) {
    j = json{{"fpga_count", t.fpga_count}, {"ctime", t.ctime}};
}

void to_json(json& j, const stack_ctype& t) {
    j = json{{"prod", t.prod}, {"conjugate", t.conjugate}};
}

void to_json(json& j, const rstack_ctype& t) {
    j = json{{"stack", t.stack}, {"conjugate", t.conjugate}};
}

void from_json(const json& j, freq_ctype& f) {
    f.centre = j.at("centre").get<double>();
    f.width = j.at("width").get<double>();
}

void from_json(const json& j, input_ctype& i) {
    i.chan_id = j.at("chan_id").get<uint32_t>();
    std::string t = j.at("correlator_input").get<std::string>();
    std::memset(i.correlator_input, 0, 32);
    t.copy(i.correlator_input, 32);
}

void from_json(const json& j, prod_ctype& p) {
    p.input_a = j.at("input_a").get<uint16_t>();
    p.input_b = j.at("input_b").get<uint16_t>();
}

void from_json(const json& j, time_ctype& t) {
    t.fpga_count = j.at("fpga_count").get<uint64_t>();
    t.ctime = j.at("ctime").get<double>();
}

void from_json(const json& j, stack_ctype& t) {
    t.prod = j.at("prod").get<uint32_t>();
    t.conjugate = j.at("conjugate").get<bool>();
}

void from_json(const json& j, rstack_ctype& t) {
    t.stack = j.at("stack").get<uint32_t>();
    t.conjugate = j.at("conjugate").get<bool>();
}

// Copy the visibility triangle out of the buffer of data, allowing for a
// possible reordering of the inputs
// TODO: port this to using map_vis_triangle. Need a unit test first.
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
            output[pi] = {(float)inputdata[2 * bi + 1], i_sign * (float)inputdata[2 * bi]};
            pi++;
        }
    }
}

// Apply a function over the visibility triangle
void map_vis_triangle(const std::vector<uint32_t>& inputmap,
    size_t block, size_t N, std::function<void(int32_t, int32_t, bool)> f
) {

    size_t pi = 0;
    uint32_t bi;
    uint32_t ii, jj;
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

            bi = prod_index(ii, jj, block, N);

            f(pi, bi, !no_flip);

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

    size_t num_elements = config.get<size_t>("/", "num_elements");

    try {
        json reorder_config = config.get<std::vector<json>>(base_path, "input_reorder");

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
