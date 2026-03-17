#include "../include/pirate/XEngineMetadata.hpp"
#include "../include/pirate/YamlFile.hpp"

#include <sstream>
#include <unordered_set>
#include <ksgpu/xassert.hpp>
#include <yaml-cpp/yaml.h>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


long XEngineMetadata::get_total_nfreq() const
{
    long ret = 0;
    for (long n: zone_nfreq)
        ret += n;
    return ret;
}


void XEngineMetadata::validate() const
{
    // Check version.
    xassert(version > 0);

    // Validate zone_nfreq and zone_freq_edges.
    xassert(zone_nfreq.size() > 0);
    xassert(zone_freq_edges.size() == zone_nfreq.size() + 1);

    for (size_t i = 0; i < zone_nfreq.size(); i++)
        xassert(zone_nfreq[i] > 0);

    for (size_t i = 0; i+1 < zone_freq_edges.size(); i++) {
        xassert(zone_freq_edges[i] > 0.0);
        xassert(zone_freq_edges[i] < zone_freq_edges[i+1]);
    }

    // Validate freq_channels (if present).
    long total_nfreq = get_total_nfreq();
    for (long ch: freq_channels) {
        if ((ch < 0) || (ch >= total_nfreq)) {
            stringstream ss;
            ss << "XEngineMetadata::validate(): freq_channels contains invalid channel "
               << ch << " (total_nfreq=" << total_nfreq << ")";
            throw runtime_error(ss.str());
        }
    }

    // Validate nbeams and beam_ids.
    xassert(nbeams > 0);

    if (beam_ids.size() > 0) {
        if (long(beam_ids.size()) != nbeams) {
            stringstream ss;
            ss << "XEngineMetadata::validate(): beam_ids.size()=" << beam_ids.size()
               << " does not match nbeams=" << nbeams;
            throw runtime_error(ss.str());
        }

        // Check for duplicate beam_ids.
        std::unordered_set<long> seen;
        for (long id : beam_ids) {
            if (!seen.insert(id).second) {
                stringstream ss;
                ss << "XEngineMetadata::validate(): duplicate beam_id " << id;
                throw runtime_error(ss.str());
            }
        }
    }

    // Validate initial_time_sample.
    xassert_ge(initial_time_sample, 0);
    xassert_divisible(initial_time_sample, 256);
}


// -------------------------------------------------------------------------------------------------


void XEngineMetadata::to_yaml(YAML::Emitter &emitter, bool verbose) const
{
    this->validate();

    emitter << YAML::BeginMap;

    // ---- Version ----

    if (verbose) {
        emitter << YAML::Comment(
            "\"X-engine metadata\" is used in two contexts:\n"
            "\n"
            " 1. Every X-engine node sends this file to every FRB node, at the beginning\n"
            "   of the TCP stream.\n"
            "\n"
            " 2. As a configuration file for the \"fake correlator\" used for testing."
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "version" << YAML::Value << version;

    // ---- Frequency channels ----

    if (verbose) {
        stringstream ss;
        ss << "Frequency channels. The observed frequency band is divided into \"zones\".\n";
        ss << "Within each zone, all frequency channels have the same width, but the\n";
        ss << "channel width may differ between zones.\n";
        ss << "  zone_nfreq: number of frequency channels in each zone.\n";
        ss << "  zone_freq_edges: frequency band edges in MHz.\n";
        ss << "For example:\n";
        ss << "  zone_nfreq: [N]      zone_freq_edges: [400,800]      one zone, channel width (400/N) MHz\n";
        ss << "  zone_nfreq: [2*N,N]  zone_freq_edges: [400,600,800]  width (100/N), (200/N) MHz in lower/upper band\n";
        ss << "\n";
        ss << "In this config, we have:\n";
        ss << "  Total frequency channels: " << get_total_nfreq() << "\n";
        ss << "  Channel widths (MHz): [ ";
        for (size_t i = 0; i < zone_nfreq.size(); i++) {
            double width = (zone_freq_edges[i+1] - zone_freq_edges[i]) / zone_nfreq[i];
            ss << (i ? ", " : "") << width;
        }
        ss << " ]";
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(ss.str()) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "zone_nfreq"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long n: zone_nfreq)
        emitter << n;
    emitter << YAML::EndSeq;

    emitter << YAML::Key << "zone_freq_edges"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (double f: zone_freq_edges)
        emitter << f;
    emitter << YAML::EndSeq;

    // ---- freq_channels (optional) ----

    if (freq_channels.size() > 0) {
        if (verbose) {
            emitter << YAML::Newline << YAML::Newline << YAML::Comment(
                "Optional: which frequency channels are present?\n"
                "A list of distinct integers 0 <= (channel_id) < (total frequency channels).\n"
                "Only makes sense in \"context 1\" (see above), to indicate which frequency channels\n"
                "are sent by a particular X-engine node."
            ) << YAML::Newline << YAML::Newline;
        }
        
        emitter << YAML::Key << "freq_channels"
                << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (long ch: freq_channels)
            emitter << ch;
        emitter << YAML::EndSeq;
    }

    // ---- Beams ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Number of beams, and their \"beam ids\" (opaque integer identifiers).\n"
            "If beam_ids is absent, it defaults to [ 0, 1, ..., (nbeams-1) ].\n"
            "Future versions of this file format will include more beam info (e.g. sky coordinates)."
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "nbeams" << YAML::Value << nbeams;

    if (beam_ids.size() > 0) {
        emitter << YAML::Key << "beam_ids"
                << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (long id: beam_ids)
            emitter << id;
        emitter << YAML::EndSeq;
    }

    // ---- initial_time_sample ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "When an X-engine node sends metadata to an FRB search node, it indicates the starting time\n"
            "of the data stream that will follow. Currently, we represent the starting time by a sample\n"
            "count, which must be a multiple of 256."
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "initial_time_sample" << YAML::Value << initial_time_sample;

    emitter << YAML::EndMap;
}


string XEngineMetadata::to_yaml_string(bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose);
    return emitter.c_str();
}


// -------------------------------------------------------------------------------------------------


// static member function
XEngineMetadata XEngineMetadata::from_yaml_string(const string &s)
{
    YAML::Node node = YAML::Load(s);
    YamlFile f("<string>", node);
    return XEngineMetadata::from_yaml(f);
}


// static member function
XEngineMetadata XEngineMetadata::from_yaml_file(const string &filename)
{
    YamlFile f(filename);
    return XEngineMetadata::from_yaml(f);
}


// static member function
XEngineMetadata XEngineMetadata::from_yaml(const YamlFile &f)
{
    XEngineMetadata ret;

    ret.version = f.get_scalar<long> ("version");
    ret.zone_nfreq = f.get_vector<long> ("zone_nfreq");
    ret.zone_freq_edges = f.get_vector<double> ("zone_freq_edges");
    ret.freq_channels = f.get_vector<long> ("freq_channels", std::vector<long>());
    ret.nbeams = f.get_scalar<long> ("nbeams");
    ret.beam_ids = f.get_vector<long> ("beam_ids", std::vector<long>());
    ret.initial_time_sample = f.get_scalar<long> ("initial_time_sample");

    // If beam_ids is absent, default to [ 0, 1, ..., (nbeams-1) ].
    if (ret.beam_ids.empty()) {
        ret.beam_ids.resize(ret.nbeams);
        for (long i = 0; i < ret.nbeams; i++)
            ret.beam_ids[i] = i;
    }

    f.check_for_invalid_keys();

    ret.validate();
    return ret;
}


// static member function
void XEngineMetadata::check_sender_consistency(const XEngineMetadata &ref, const XEngineMetadata &m)
{
    if (m.zone_nfreq != ref.zone_nfreq) {
        stringstream ss;
        ss << "XEngineMetadata::check_sender_consistency: mismatch in zone_nfreq";
        throw runtime_error(ss.str());
    }
    if (m.zone_freq_edges.size() != ref.zone_freq_edges.size()) {
        stringstream ss;
        ss << "XEngineMetadata::check_sender_consistency: mismatch in zone_freq_edges (different sizes)";
        throw runtime_error(ss.str());
    }
    for (size_t i = 0; i < m.zone_freq_edges.size(); i++) {
        double diff = m.zone_freq_edges[i] - ref.zone_freq_edges[i];
        if ((diff < -1.0e-3) || (diff > 1.0e-3)) {
            stringstream ss;
            ss << "XEngineMetadata::check_sender_consistency: mismatch in zone_freq_edges[" << i << "]: got "
               << m.zone_freq_edges[i] << ", expected " << ref.zone_freq_edges[i];
            throw runtime_error(ss.str());
        }
    }
    if (m.nbeams != ref.nbeams) {
        stringstream ss;
        ss << "XEngineMetadata::check_sender_consistency: mismatch in nbeams: got " << m.nbeams
           << ", expected " << ref.nbeams;
        throw runtime_error(ss.str());
    }
    if (m.beam_ids != ref.beam_ids) {
        stringstream ss;
        ss << "XEngineMetadata::check_sender_consistency: mismatch in beam_ids";
        throw runtime_error(ss.str());
    }
}


}  // namespace pirate
