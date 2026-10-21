#include "interfaceRegistry.hpp"

#include "fmt.hpp" // for format, fmt

#include <map>       // for map
#include <stdexcept> // for invalid_argument, runtime_error
#include <stdio.h>   // for snprintf

using json = nlohmann::json;

namespace kotekan {

namespace {

struct Registration {
    int64_t schema;
    json (*builder)(); // nullptr once the name is registered twice
};

std::map<std::string, Registration>& registry() {
    static std::map<std::string, Registration> registrations;
    return registrations;
}

bool is_printable_ascii(const std::string& s) {
    for (char c : s)
        if (c < 0x20 || c > 0x7e)
            return false;
    return true;
}

// Error messages go into /version, whose dump throws on invalid UTF-8.
std::string to_printable_ascii(std::string s) {
    for (char& c : s)
        if (c < 0x20 || c > 0x7e)
            c = '?';
    return s;
}

// path is the JSON pointer of value, for the error message.
void check_canonical(const json& value, const std::string& path) {
    switch (value.type()) {
        case json::value_t::object:
            for (auto& it : value.items()) {
                if (!is_printable_ascii(it.key()))
                    throw std::invalid_argument(
                        fmt::format(fmt("key in '{:s}' is not printable ASCII"), path));
                check_canonical(it.value(), path + "/" + it.key());
            }
            break;
        case json::value_t::array:
            for (size_t i = 0; i < value.size(); i++)
                check_canonical(value[i], path + "/" + std::to_string(i));
            break;
        case json::value_t::string:
            if (!is_printable_ascii(value.get_ref<const std::string&>()))
                throw std::invalid_argument(
                    fmt::format(fmt("string at '{:s}' is not printable ASCII"), path));
            break;
        case json::value_t::number_unsigned:
            if (value.get<uint64_t>() > INT64_MAX)
                throw std::invalid_argument(
                    fmt::format(fmt("integer at '{:s}' does not fit in an int64"), path));
            break;
        case json::value_t::boolean:
        case json::value_t::number_integer:
            break;
        case json::value_t::number_float:
            throw std::invalid_argument(
                fmt::format(fmt("float at '{:s}'; write doubles with canonical_double()"), path));
        default:
            throw std::invalid_argument(
                fmt::format(fmt("{:s} at '{:s}' is not allowed"), value.type_name(), path));
    }
}

json build_interface(const std::string& name, const Registration& registration) {
    if (registration.builder == nullptr)
        throw std::runtime_error("interface registered more than once");
    json sections = registration.builder();
    if (!sections.is_object())
        throw std::invalid_argument("builder must return an object of sections");

    // The whole document is checked first, so an error names its full path.
    json doc = {{"interface", name}, {"schema", registration.schema}, {"sections", sections}};
    std::string digest = fnv1a64_digest(canonical_json(doc));

    json section_digests = json::object();
    for (auto& it : sections.items())
        section_digests[it.key()] = fnv1a64_digest(canonical_json(it.value()));

    return {{"schema", registration.schema},
            {"digest", digest},
            {"sections", section_digests},
            {"doc", doc}};
}

} // namespace

int register_interface(const std::string& name, int64_t schema, json (*builder)()) {
    auto& r = registry();
    if (r.count(name))
        r[name].builder = nullptr;
    else
        r[name] = {schema, builder};
    return 0;
}

json build_interfaces() {
    json interfaces = json::object();
    for (auto& [name, registration] : registry()) {
        try {
            interfaces[name] = build_interface(name, registration);
        } catch (const std::exception& e) {
            interfaces[name] = {{"error", to_printable_ascii(e.what())}};
        } catch (...) {
            interfaces[name] = {{"error", "builder threw a non-standard exception"}};
        }
    }
    return interfaces;
}

std::string canonical_json(const json& value) {
    check_canonical(value, "");
    return value.dump(-1, ' ', true);
}

std::string canonical_double(double value) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.17g", value);
    return buf;
}

std::string fnv1a64_digest(const std::string& bytes) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (unsigned char c : bytes) {
        hash ^= c;
        hash *= 0x100000001b3ULL;
    }
    return fmt::format(fmt("fnv1a64:{:016x}"), hash);
}

} // namespace kotekan
