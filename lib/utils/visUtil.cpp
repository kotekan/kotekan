#include "visUtil.hpp"

#include "Config.hpp" // for Config

#include <cstring>   // for memset
#include <exception> // for exception
#include <iterator>  // for back_insert_iterator, back_inserter
#include <limits>
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

double SlidingWindowMinMax::get_min() {
    return min_deque.front();
}

double SlidingWindowMinMax::get_max() {
    return max_deque.front();
}

void SlidingWindowMinMax::add_tail(double val) {
    while (!min_deque.empty() && val < min_deque.back()) {
        min_deque.pop_back();
    }
    min_deque.push_back(val);

    while (!max_deque.empty() && val > max_deque.back()) {
        max_deque.pop_back();
    }
    max_deque.push_back(val);
}

void SlidingWindowMinMax::remove_head(double val) {
    if (val == min_deque.front())
        min_deque.pop_front();

    if (val == max_deque.front())
        max_deque.pop_front();
}

StatTracker::StatTracker(std::string name, std::string unit, size_t size, bool is_optimized) :
    rbuf(std::make_unique<sample[]>(size)),
    end(0),
    buf_size(size),
    count(0),
    avg(0),
    dist(0),
    var(0),
    std_dev(0),
    name(name),
    unit(unit),
    is_optimized(is_optimized){};

void StatTracker::add_sample(double new_val) {
    std::lock_guard<std::mutex> lock(tracker_lock);

    double old_val = rbuf[end].value;
    rbuf[end].value = new_val;
    rbuf[end].timestamp = std::chrono::system_clock::now();
    end = (end + 1) % buf_size;
    if (is_optimized)
        min_max.add_tail(new_val);

    if (count < buf_size) {
        double old_avg = avg;
        avg += (new_val - old_avg) / (++count);
        dist += (new_val - avg) * (new_val - old_avg);
        var = (count <= 1) ? NAN : dist / (count - 1);
    } else {
        double old_avg = avg;
        if (is_optimized)
            min_max.remove_head(old_val);
        avg = old_avg + (new_val - old_val) / buf_size;
        var += (new_val - old_val) * (new_val - avg + old_val - old_avg) / (buf_size - 1);
    }

    std_dev = sqrt(var);
}

double StatTracker::get_max() {
    std::lock_guard<std::mutex> lock(tracker_lock);

    if (count == 0) {
        return NAN;
    }

    if (is_optimized) {
        return min_max.get_max();
    } else {
        // brute force way to get max
        double max = std::numeric_limits<double>::lowest();
        size_t size = std::min(count, buf_size);
        for (size_t i = 0; i < size; i++)
            max = std::max(max, rbuf[i].value);

        return max;
    }
}

double StatTracker::get_min() {
    std::lock_guard<std::mutex> lock(tracker_lock);

    if (count == 0) {
        return NAN;
    }

    if (is_optimized) {
        return min_max.get_min();
    } else {
        // brute force way to get min
        double min = std::numeric_limits<double>::max();
        size_t size = std::min(count, buf_size);
        for (size_t i = 0; i < size; i++)
            min = std::min(min, rbuf[i].value);

        return min;
    }
}

double StatTracker::get_avg() {
    std::lock_guard<std::mutex> lock(tracker_lock);

    if (count == 0) {
        return NAN;
    }
    return avg;
}

double StatTracker::get_std_dev() {
    std::lock_guard<std::mutex> lock(tracker_lock);

    if (count <= 1) {
        return NAN;
    }
    return std_dev;
}

nlohmann::json StatTracker::get_json() {
    std::lock_guard<std::mutex> lock(tracker_lock);

    nlohmann::json tracker_json = {};
    tracker_json["unit"] = unit;
    for (size_t i = 0; i < count; i++) {
        nlohmann::json sample_json = {};
        sample_json["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       rbuf[i].timestamp.time_since_epoch())
                                       .count();
        sample_json["value"] = rbuf[i].value;
        tracker_json["samples"].push_back(sample_json);
    }

    return tracker_json;
}

nlohmann::json StatTracker::get_current_json() {
    nlohmann::json tracker_json = {};

    tracker_json["unit"] = unit;
    size_t ind = (end + buf_size - 1) % buf_size;
    tracker_json["cur"]["value"] = count ? rbuf[ind].value : NAN;
    tracker_json["cur"]["timestamp"] = count
                                           ? std::chrono::duration_cast<std::chrono::milliseconds>(
                                                 rbuf[ind].timestamp.time_since_epoch())
                                                 .count()
                                           : NAN;
    tracker_json["min"] = get_min();
    tracker_json["max"] = get_max();
    tracker_json["avg"] = get_avg();
    tracker_json["std"] = get_std_dev();

    return tracker_json;
}

std::vector<std::string> regex_split(const std::string input, const std::string reg) {
    std::vector<std::string> split_array;
    std::regex split_regex(reg);
    std::copy(std::sregex_token_iterator(input.begin(), input.end(), split_regex, -1),
              std::sregex_token_iterator(), std::back_inserter(split_array));
    return split_array;
}
