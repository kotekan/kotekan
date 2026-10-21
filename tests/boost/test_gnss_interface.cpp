#define BOOST_TEST_MODULE "test_gnss_interface"

#include "gnssSignal.hpp"        // for ALL_SIGNALS, SignalDescriptor, signal_by_name
#include "interfaceRegistry.hpp" // for build_interfaces, canonical_json, fnv1a64_digest

#include "json.hpp" // for json

#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK_MESSAGE, BOOST_CHECK_EQUAL, ...
#include <fstream>                           // for ifstream
#include <iterator>                          // for size
#include <regex>                             // for regex, sregex_iterator
#include <set>                               // for set
#include <sstream>                           // for stringstream
#include <stdexcept>                         // for runtime_error
#include <string>                            // for string

using json = nlohmann::json;
using kotekan::canonical_json;
using kotekan::fnv1a64_digest;

namespace {

std::string read_source(const std::string& name) {
    const std::string path = std::string(GNSS_SOURCE_DIR) + "/" + name;
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// The "gnss" entry of build_interfaces(): schema, digests and the document.
json gnss_entry() {
    json all = kotekan::build_interfaces();
    BOOST_REQUIRE(all.contains("gnss"));
    BOOST_REQUIRE_MESSAGE(!all["gnss"].contains("error"), all["gnss"].dump());
    return all["gnss"];
}

// Every object key in j, at any depth.
void collect_keys(const json& j, std::set<std::string>& keys) {
    if (j.is_object())
        for (auto& it : j.items()) {
            keys.insert(it.key());
            collect_keys(it.value(), keys);
        }
    else if (j.is_array())
        for (auto& v : j)
            collect_keys(v, keys);
}

} // namespace

BOOST_AUTO_TEST_CASE(matches_golden) {
    const json entry = gnss_entry();
    const json golden = json::parse(read_source("gnss_interface.json"));
    const std::string bytes = canonical_json(entry["doc"]);
    BOOST_CHECK_MESSAGE(
        bytes == canonical_json(golden),
        "the gnss interface document differs from gnss_interface.json; regenerate "
        "it as gnssInterface.hpp says. Differences as a JSON patch from the golden: "
            << json::diff(golden, entry["doc"]).dump());
    BOOST_CHECK_EQUAL(entry["schema"].get<int>(), 1);
    BOOST_CHECK_EQUAL(entry["digest"].get<std::string>(), fnv1a64_digest(canonical_json(golden)));
    for (auto& it : golden["sections"].items())
        BOOST_CHECK_EQUAL(entry["sections"][it.key()].get<std::string>(),
                          fnv1a64_digest(canonical_json(it.value())));
    BOOST_TEST_MESSAGE("gnss " << entry["digest"].get<std::string>() << " "
                               << entry["sections"].dump());
}

BOOST_AUTO_TEST_CASE(deterministic) {
    const json a = gnss_entry();
    const json b = gnss_entry();
    BOOST_CHECK_EQUAL(canonical_json(a["doc"]), canonical_json(b["doc"]));
    BOOST_CHECK_EQUAL(a["digest"].get<std::string>(), b["digest"].get<std::string>());
}

BOOST_AUTO_TEST_CASE(signals_resolve) {
    const json signals = gnss_entry()["doc"]["sections"]["signals"];
    for (auto& it : signals.items()) {
        const gnss::SignalDescriptor* s = gnss::signal_by_name(it.key());
        BOOST_CHECK_MESSAGE(s != nullptr && it.key() == s->name,
                            it.key() << " does not resolve through signal_by_name");
    }
    for (const gnss::SignalDescriptor* s : gnss::ALL_SIGNALS)
        BOOST_CHECK_MESSAGE(signals.contains(s->name), s->name << " is not in the document");
    // A name listed twice in ALL_SIGNALS shows up as one key short.
    BOOST_CHECK_EQUAL(signals.size(), std::size(gnss::ALL_SIGNALS));
}

// The headers against the document: every descriptor defined in gnssSignal.hpp must be in
// ALL_SIGNALS, and every constant with a layout prefix must be a key in the document.
BOOST_AUTO_TEST_CASE(headers_covered) {
    std::set<std::string> listed;
    for (const gnss::SignalDescriptor* s : gnss::ALL_SIGNALS)
        listed.insert(s->name);
    const std::string sig = read_source("gnssSignal.hpp");
    const std::regex descriptor(R"(inline\s+constexpr\s+SignalDescriptor\s+(\w+)\s*=)");
    size_t n_desc = 0;
    for (std::sregex_iterator m(sig.begin(), sig.end(), descriptor), end; m != end; ++m) {
        ++n_desc;
        BOOST_CHECK_MESSAGE(listed.count((*m)[1]), (*m)[1] << " in gnssSignal.hpp is not in "
                                                              "ALL_SIGNALS: a config naming it "
                                                              "cannot start a node");
    }
    // Also fails when a descriptor is defined in a form this scan does not see.
    BOOST_CHECK_EQUAL(n_desc, std::size(gnss::ALL_SIGNALS));

    std::set<std::string> keys;
    collect_keys(gnss_entry()["doc"], keys);
    const std::regex constant(
        R"(constexpr[^=;(){}]*?\b((?:REC|RECORD|CHAN|ELEM|CMB|TELEM|CUBE|ROW|ROWS|LANES)_\w+)\s*=)");
    size_t n_const = 0;
    for (const char* header :
         {"gnssSignal.hpp", "gnssRecord.hpp", "gnssTelem.hpp", "gnssGpuChain.hpp"}) {
        const std::string text = read_source(header);
        for (std::sregex_iterator m(text.begin(), text.end(), constant), end; m != end; ++m) {
            ++n_const;
            BOOST_CHECK_MESSAGE(keys.count((*m)[1]), (*m)[1] << " in " << header
                                                             << " is not in the gnss interface "
                                                                "document (gnssInterface.cpp)");
        }
    }
    // A scan that finds nothing would pass anything.
    BOOST_CHECK_GT(n_const, 50u);
}
