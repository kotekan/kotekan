#include "visUtil.hpp"

#include "Config.hpp" // for Config

#include <cstring>   // for memset
#include <exception> // for exception
#include <iterator>  // for back_insert_iterator, back_inserter
#include <regex>     // for sregex_token_iterator, match_results<>::_Base_type, _NFA, regex
#include <sstream>   // for basic_stringbuf<>::int_type, basic_stringbuf<>::pos_type, basic_st...
#include <stdexcept> // for runtime_error, invalid_argument

using nlohmann::json;

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

bool operator!=(const rstack_ctype& lhs, const rstack_ctype& rhs) {
    return (lhs.stack != rhs.stack) || (lhs.conjugate != rhs.conjugate);
}

// JSON converters
void to_json(json& j, const freq_ctype& f) {
    j = json{{"centre", f.centre}, {"width", f.width}};
}

void to_json(json& j, const input_ctype& i) {
    j = json{i.chan_id, i.correlator_input};
}

void to_json(json& j, const prod_ctype& p) {
    j = json{p.input_a, p.input_b};
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
    i.chan_id = j.at(0).get<uint32_t>();
    std::string t = j.at(1).get<std::string>();
    std::memset(i.correlator_input, 0, 32);
    t.copy(i.correlator_input, 32);
}

void from_json(const json& j, prod_ctype& p) {
    p.input_a = j.at(0).get<uint16_t>();
    p.input_b = j.at(1).get<uint16_t>();
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

std::string json_type_name(nlohmann::json& value) {
    switch (value.type()) {
        case (json::value_t::number_integer):
            return "integer";
        case (json::value_t::number_unsigned):
            return "integer";
        case (json::value_t::number_float):
            return "float";
        default:
            return value.type_name();
    }
}

// Copy the visibility triangle out of the buffer of data, allowing for a
// possible reordering of the inputs
// TODO: port this to using map_vis_triangle. Need a unit test first.
void copy_vis_triangle(const int32_t* inputdata, const std::vector<uint32_t>& inputmap,
                       size_t block, size_t N, gsl::span<cfloat> output) {

    auto copyfunc = [&](int32_t pi, int32_t bi, bool conj) {
        int i_sign = conj ? -1 : 1;
        output[pi] = {(float)inputdata[2 * bi + 1], i_sign * (float)inputdata[2 * bi]};
    };

    map_vis_triangle(inputmap, block, N, 0, copyfunc);
}

// Apply a function over the visibility triangle
void map_vis_triangle(const std::vector<uint32_t>& inputmap, size_t block, size_t N, uint32_t freq,
                      std::function<void(int32_t, int32_t, bool)> f) {

    size_t pi = 0;
    uint32_t bi;
    uint32_t ii, jj;
    bool no_flip;

    if (*std::max_element(inputmap.begin(), inputmap.end()) >= N) {
        throw std::invalid_argument("Input map asks for elements out of range.");
    }

    uint32_t offset = freq * gpu_N2_size(N, block);

    for (auto i = inputmap.begin(); i != inputmap.end(); i++) {
        for (auto j = i; j != inputmap.end(); j++) {

            // Account for the case when the reordering means we should be
            // indexing into the lower triangle, by flipping into the upper
            // triangle and conjugating.
            no_flip = *i <= *j;
            ii = no_flip ? *i : *j;
            jj = no_flip ? *j : *i;

            bi = offset + prod_index(ii, jj, block, N);

            f(pi, bi, !no_flip);

            pi++;
        }
    }
}


std::tuple<uint32_t, uint32_t, std::string> parse_reorder_single(json j) {
    if (!j.is_array() || j.size() != 3) {
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

    if (!j.is_array()) {
        throw std::runtime_error("Was expecting list of input orders.");
    }

    for (auto& element : j) {
        std::tie(adc_id, chan_id, serial) = parse_reorder_single(element);

        adc_ids.push_back(adc_id);
        inputmap.emplace_back(chan_id, serial);
    }

    return std::make_tuple(adc_ids, inputmap);
}

std::tuple<std::vector<uint32_t>, std::vector<input_ctype>> default_reorder(size_t num_elements) {

    std::vector<uint32_t> adc_ids;
    std::vector<input_ctype> inputmap;

    for (uint32_t i = 0; i < num_elements; i++) {
        adc_ids.push_back(i);
        inputmap.emplace_back(i, "INVALID");
    }

    return std::make_tuple(adc_ids, inputmap);
}

std::tuple<std::vector<uint32_t>, std::vector<input_ctype>>
parse_reorder_default(kotekan::Config& config, const std::string base_path) {

    size_t num_elements = config.get<size_t>("/", "num_elements");

    try {
        json reorder_config = config.get<std::vector<json>>(base_path, "input_reorder");

        return parse_reorder(reorder_config);
    } catch (const std::exception& e) {
        return default_reorder(num_elements);
    }
}


size_t _member_alignment(size_t offset, size_t size) {
    return (((size - (offset % size)) % size) + offset);
}


movingAverage::movingAverage(double length) {
    // Calculate the coefficient for the moving average as a halving of the weight
    alpha = 1.0 - pow(2, -1.0 / length);
}


void movingAverage::add_sample(double value) {

    // Special case for the first sample.
    if (!initialised) {
        current_value = value;
        initialised = true;
    } else {
        current_value = alpha * value + (1 - alpha) * current_value;
    }
}

double movingAverage::average() {
    if (!initialised) {
        return NAN;
    }
    return current_value;
}

sampleBuffer::sampleBuffer(size_t size) :
    rbuf(std::make_unique<double[]>(size)),
    front(0),
    end(0),
    buf_size(size),
    count(0){};

void sampleBuffer::add_sample(double sample) {
    rbuf[end] = sample;
    if (count < buf_size) {
        count++;
    }

    if (count == buf_size) {
        front = (front + 1) % buf_size;
    }

    end = (end + 1) % buf_size;
}

double sampleBuffer::get_max() {
    double max = std::numeric_limits<double>::min();
    size_t index;

    if (count == 0) {
        return 0.0;
    }

    for (size_t i = 0; i < count; i++) {
        index = (front + i) % buf_size;
        if (max < rbuf[index]) {
            max = rbuf[index];
        }
    }

    return max;
}

double sampleBuffer::get_min() {
    double min = std::numeric_limits<double>::max();
    size_t index;

    if (count == 0) {
        return 0.0;
    }

    for (size_t i = 0; i < count; i++) {
        index = (front + i) % buf_size;
        if (min > rbuf[index]) {
            min = rbuf[index];
        }
    }

    return min;
}

double sampleBuffer::get_avg() {
    double sum = 0.0;
    size_t index;

    if (count == 0) {
        return 0.0;
    }

    for (size_t i = 0; i < count; i++) {
        index = (front + i) % buf_size;
        sum += rbuf[index];
    }

    return sum / count;
}

double sampleBuffer::get_std_dev() {
    double standardDeviation = 0.0;
    double mean;
    size_t index;

    if (count == 0) {
        return 0.0;
    }

    mean = this->get_avg();

    for (size_t i = 0; i < count; i++) {
        index = (front + i) % buf_size;
        standardDeviation += pow(rbuf[index] - mean, 2);
    }

    return sqrt(standardDeviation / count);
}

std::vector<std::string> regex_split(const std::string input, const std::string reg) {
    std::vector<std::string> split_array;
    std::regex split_regex(reg);
    std::copy(std::sregex_token_iterator(input.begin(), input.end(), split_regex, -1),
              std::sregex_token_iterator(), std::back_inserter(split_array));
    return split_array;
}
