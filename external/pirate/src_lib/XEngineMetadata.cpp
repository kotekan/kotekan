#include "../include/pirate/XEngineMetadata.hpp"
#include "../include/pirate/YamlFile.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <ksgpu/rand_utils.hpp>
#include <ksgpu/xassert.hpp>
#include <yaml-cpp/yaml.h>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Tolerances used by validate() and check_sender_consistency().
static constexpr double cosine_eps = 1.0e-9;     // direction-cosine sanity / equality
static constexpr double unit_vec_eps = 1.0e-6;   // |v|^2 - 1 tolerance
static constexpr double angle_eps = 1.0e-6;      // lat/lon/coelev equality
static constexpr double freq_eps = 1.0e-3;       // MHz, zone_freq_edges
static constexpr double dish_sep_eps = 1.0e-6;   // meters, dish separations
static constexpr double variance_eps = 1.0e-12;  // noise_variance equality


// Throws if x is NaN or +/-Inf.
static void _check_finite(double x, const char *name)
{
    if (!std::isfinite(x)) {
        stringstream ss;
        ss << "XEngineMetadata::validate(): " << name << " = " << x << " is not finite";
        throw runtime_error(ss.str());
    }
}


// Throws if any element of 'v' is NaN or +/-Inf. Templated so std::vector<double> and std::array<double, N>
// (and any other container with size()/operator[]/double-convertible elements) all work.
template<typename Vec>
static void _check_finite(const Vec &v, const char *name)
{
    for (size_t i = 0; i < v.size(); i++) {
        if (!std::isfinite(v[i])) {
            stringstream ss;
            ss << "XEngineMetadata::validate(): " << name << "[" << i << "] = " << v[i] << " is not finite";
            throw runtime_error(ss.str());
        }
    }
}


static void _check_unit_vector(const std::array<double, 3> &v, const char *name)
{
    _check_finite(v, name);

    double norm2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    if (std::abs(norm2 - 1.0) > unit_vec_eps) {
        stringstream ss;
        ss << "XEngineMetadata::validate(): " << name << " is not a unit vector (|v|^2 = " << norm2 << ")";
        throw runtime_error(ss.str());
    }
}


// Throws if the two unit vectors are not orthogonal (dot product within unit_vec_eps of 0).
// Caller should verify each is finite (via _check_unit_vector) before calling.
static void _check_orthogonal(const std::array<double, 3> &a, const std::array<double, 3> &b,
                              const char *name_a, const char *name_b)
{
    double dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    if (std::abs(dot) > unit_vec_eps) {
        stringstream ss;
        ss << "XEngineMetadata::validate(): " << name_a << " and " << name_b
           << " are not orthogonal (dot product = " << dot << ")";
        throw runtime_error(ss.str());
    }
}


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
    xassert_eq(version, 2L);

    // Validate zone_nfreq and zone_freq_edges.
    xassert(zone_nfreq.size() > 0);
    xassert(zone_freq_edges.size() == zone_nfreq.size() + 1);

    for (size_t i = 0; i < zone_nfreq.size(); i++)
        xassert(zone_nfreq[i] > 0);

    _check_finite(zone_freq_edges, "zone_freq_edges");
    for (size_t i = 0; i+1 < zone_freq_edges.size(); i++) {
        xassert(zone_freq_edges[i] > 0.0);
        xassert(zone_freq_edges[i] < zone_freq_edges[i+1]);
    }

    // Validate freq_channels (if present): each value in range, no duplicates.
    long total_nfreq = get_total_nfreq();
    std::unordered_set<long> seen_channels;
    for (long ch: freq_channels) {
        if ((ch < 0) || (ch >= total_nfreq)) {
            stringstream ss;
            ss << "XEngineMetadata::validate(): freq_channels contains invalid channel "
               << ch << " (total_nfreq=" << total_nfreq << ")";
            throw runtime_error(ss.str());
        }
        if (!seen_channels.insert(ch).second) {
            stringstream ss;
            ss << "XEngineMetadata::validate(): duplicate freq_channel " << ch;
            throw runtime_error(ss.str());
        }
    }

    // Validate beam_ids (must be present, unique).
    long nbeams = long(beam_ids.size());
    xassert(nbeams > 0);

    std::unordered_set<long> seen;
    for (long id : beam_ids) {
        if (!seen.insert(id).second) {
            stringstream ss;
            ss << "XEngineMetadata::validate(): duplicate beam_id " << id;
            throw runtime_error(ss.str());
        }
    }

    // Validate beam_positions_x / beam_positions_y.
    if (long(beam_positions_x.size()) != nbeams) {
        stringstream ss;
        ss << "XEngineMetadata::validate(): beam_positions_x.size()=" << beam_positions_x.size()
           << " does not match nbeams=" << nbeams;
        throw runtime_error(ss.str());
    }
    if (long(beam_positions_y.size()) != nbeams) {
        stringstream ss;
        ss << "XEngineMetadata::validate(): beam_positions_y.size()=" << beam_positions_y.size()
           << " does not match nbeams=" << nbeams;
        throw runtime_error(ss.str());
    }
    _check_finite(beam_positions_x, "beam_positions_x");
    _check_finite(beam_positions_y, "beam_positions_y");
    for (long i = 0; i < nbeams; i++) {
        double bx = beam_positions_x[i];
        double by = beam_positions_y[i];
        if (bx*bx + by*by > 1.0 + cosine_eps) {
            stringstream ss;
            ss << "XEngineMetadata::validate(): beam " << i << " has direction cosines outside the unit disk: ("
               << bx << ", " << by << ")";
            throw runtime_error(ss.str());
        }
    }

    // Validate timekeeping.
    xassert_gt(unix_ns_at_seq_0, 0L);
    xassert_gt(dt_ns_per_seq, 0L);
    xassert_gt(seq_per_frb_time_sample, 0L);

    // Validate telescope params.
    xassert_ge(tel_origin_itrs_lat_deg, -90.0);
    xassert_le(tel_origin_itrs_lat_deg,  90.0);
    xassert_ge(tel_origin_itrs_lon_deg, -180.0);
    xassert_le(tel_origin_itrs_lon_deg,  360.0);

    _check_unit_vector(tel_grid_x_axis,    "tel_grid_x_axis");
    _check_unit_vector(tel_grid_y_axis,    "tel_grid_y_axis");
    _check_unit_vector(tel_dish_elev_axis, "tel_dish_elev_axis");
    _check_unit_vector(tel_dish_vert_axis, "tel_dish_vert_axis");

    _check_orthogonal(tel_grid_x_axis,    tel_grid_y_axis,    "tel_grid_x_axis",    "tel_grid_y_axis");
    _check_orthogonal(tel_dish_elev_axis, tel_dish_vert_axis, "tel_dish_elev_axis", "tel_dish_vert_axis");

    xassert_ge(tel_dish_coelev_deg, -90.0);
    xassert_le(tel_dish_coelev_deg,  90.0);

    _check_finite(tel_dish_separation_x_m, "tel_dish_separation_x_m");
    _check_finite(tel_dish_separation_y_m, "tel_dish_separation_y_m");
    xassert_gt(tel_dish_separation_x_m, 0.0);
    xassert_gt(tel_dish_separation_y_m, 0.0);

    // Validate noise_variance.
    if (long(noise_variance.size()) != long(zone_nfreq.size())) {
        stringstream ss;
        ss << "XEngineMetadata::validate(): noise_variance.size()=" << noise_variance.size()
           << " does not match nzones=" << zone_nfreq.size();
        throw runtime_error(ss.str());
    }
    _check_finite(noise_variance, "noise_variance");
    for (size_t i = 0; i < noise_variance.size(); i++) {
        if (!(noise_variance[i] > 0.0)) {
            stringstream ss;
            ss << "XEngineMetadata::validate(): noise_variance[" << i << "]=" << noise_variance[i]
               << " must be positive";
            throw runtime_error(ss.str());
        }
    }
}


// -------------------------------------------------------------------------------------------------


// Helper: emit a length-3 unit vector as a flow-style sequence of doubles.
static void _emit_axis(YAML::Emitter &emitter, const char *key, const std::array<double, 3> &v)
{
    emitter << YAML::Key << key
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (double x: v)
        emitter << x;
    emitter << YAML::EndSeq;
}


void XEngineMetadata::to_yaml(YAML::Emitter &emitter, bool verbose) const
{
    this->validate();
    long nbeams = get_nbeams();

    emitter << YAML::BeginMap;

    // ---- Header / version ----

    if (verbose) {
        emitter << YAML::Comment(
            "\"X-engine metadata\" is used in two contexts:\n"
            "\n"
            " 1. Every X-engine node sends this file to every FRB node, at the beginning\n"
            "   of the TCP stream.\n"
            "\n"
            " 2. As a configuration file for the \"fake X-engine\" used for testing."
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
            "Beams: an integer 'beamset' identifier, plus per-beam id and position.\n"
            "  beam_ids:         length nbeams, integer ids.\n"
            "  beam_positions_x: length nbeams, direction cosine b.x in grid frame.\n"
            "  beam_positions_y: length nbeams, direction cosine b.y in grid frame."
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "beamset" << YAML::Value << beamset;

    emitter << YAML::Key << "beam_ids"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long id: beam_ids)
        emitter << id;
    emitter << YAML::EndSeq;

    emitter << YAML::Key << "beam_positions_x"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (double x: beam_positions_x)
        emitter << x;
    emitter << YAML::EndSeq;

    emitter << YAML::Key << "beam_positions_y"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (double y: beam_positions_y)
        emitter << y;
    emitter << YAML::EndSeq;

    (void) nbeams;  // currently unused outside validate(); silence unused-warn

    // ---- Timekeeping ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Timekeeping: the X-engine uses an FPGA seq number to track time.\n"
            "  unix_ns_at_seq_0:        UNIX nanoseconds at seq=0.\n"
            "  dt_ns_per_seq:           nanoseconds per seq tick.\n"
            "  seq_per_frb_time_sample: seq ticks per FRB time sample."
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "unix_ns_at_seq_0" << YAML::Value << unix_ns_at_seq_0;
    emitter << YAML::Key << "dt_ns_per_seq" << YAML::Value << dt_ns_per_seq;
    emitter << YAML::Key << "seq_per_frb_time_sample" << YAML::Value << seq_per_frb_time_sample;

    // ---- Telescope ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Telescope alignment / localization parameters."
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "tel_origin_itrs_lat_deg" << YAML::Value << tel_origin_itrs_lat_deg;
    emitter << YAML::Key << "tel_origin_itrs_lon_deg" << YAML::Value << tel_origin_itrs_lon_deg;

    _emit_axis(emitter, "tel_grid_x_axis",    tel_grid_x_axis);
    _emit_axis(emitter, "tel_grid_y_axis",    tel_grid_y_axis);
    _emit_axis(emitter, "tel_dish_elev_axis", tel_dish_elev_axis);
    _emit_axis(emitter, "tel_dish_vert_axis", tel_dish_vert_axis);

    emitter << YAML::Key << "tel_dish_coelev_deg" << YAML::Value << tel_dish_coelev_deg;
    emitter << YAML::Key << "tel_dish_separation_x_m" << YAML::Value << tel_dish_separation_x_m;
    emitter << YAML::Key << "tel_dish_separation_y_m" << YAML::Value << tel_dish_separation_y_m;

    // ---- Noise variance ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Per-frequency-zone noise variance, length nzones (=len(zone_nfreq)).\n"
            "Temporary kludge: the FRB server assumes mean-zero, time-uncorrelated\n"
            "noise with this variance. Will be generalized in a future revision."
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "noise_variance"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (double v: noise_variance)
        emitter << v;
    emitter << YAML::EndSeq;

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

    ret.beamset = f.get_scalar<long> ("beamset");
    ret.beam_ids = f.get_vector<long> ("beam_ids");
    ret.beam_positions_x = f.get_vector<double> ("beam_positions_x");
    ret.beam_positions_y = f.get_vector<double> ("beam_positions_y");

    ret.unix_ns_at_seq_0 = f.get_scalar<long> ("unix_ns_at_seq_0");
    ret.dt_ns_per_seq = f.get_scalar<long> ("dt_ns_per_seq");
    ret.seq_per_frb_time_sample = f.get_scalar<long> ("seq_per_frb_time_sample");

    ret.tel_origin_itrs_lat_deg = f.get_scalar<double> ("tel_origin_itrs_lat_deg");
    ret.tel_origin_itrs_lon_deg = f.get_scalar<double> ("tel_origin_itrs_lon_deg");
    ret.tel_grid_x_axis = f.get_array<double, 3> ("tel_grid_x_axis");
    ret.tel_grid_y_axis = f.get_array<double, 3> ("tel_grid_y_axis");
    ret.tel_dish_elev_axis = f.get_array<double, 3> ("tel_dish_elev_axis");
    ret.tel_dish_vert_axis = f.get_array<double, 3> ("tel_dish_vert_axis");
    ret.tel_dish_coelev_deg = f.get_scalar<double> ("tel_dish_coelev_deg");
    ret.tel_dish_separation_x_m = f.get_scalar<double> ("tel_dish_separation_x_m");
    ret.tel_dish_separation_y_m = f.get_scalar<double> ("tel_dish_separation_y_m");

    // noise_variance can be either a scalar (broadcast to all zones) or a sequence of length nzones.
    YamlFile nv_node = f["noise_variance"];
    long nzones = long(ret.zone_nfreq.size());
    if (nv_node.type() == YAML::NodeType::Scalar) {
        double v = nv_node.as_scalar<double>();
        ret.noise_variance.assign(nzones, v);
    } else {
        ret.noise_variance = nv_node.as_vector<double>();
    }

    f.check_for_invalid_keys();

    ret.validate();
    return ret;
}


// -------------------------------------------------------------------------------------------------


// Helper: throw a check_sender_consistency mismatch with a formatted message.
[[noreturn]] static void _consistency_throw(const string &msg)
{
    throw runtime_error("XEngineMetadata::check_sender_consistency: " + msg);
}


static void _check_double_eq(double a, double b, double eps, const char *name)
{
    if (std::abs(a - b) > eps) {
        stringstream ss;
        ss << "mismatch in " << name << ": got " << a << ", expected " << b;
        _consistency_throw(ss.str());
    }
}


// Templated so both std::vector<double> and std::array<double, N> work.
template<typename Vec>
static void _check_double_vec_eq(const Vec &a, const Vec &b, double eps, const char *name)
{
    if (a.size() != b.size()) {
        stringstream ss;
        ss << "mismatch in " << name << " (different sizes: " << a.size() << " vs " << b.size() << ")";
        _consistency_throw(ss.str());
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > eps) {
            stringstream ss;
            ss << "mismatch in " << name << "[" << i << "]: got " << a[i] << ", expected " << b[i];
            _consistency_throw(ss.str());
        }
    }
}


// static member function
void XEngineMetadata::check_sender_consistency(const XEngineMetadata &ref, const XEngineMetadata &m)
{
    if (m.version != ref.version) {
        stringstream ss;
        ss << "mismatch in version: got " << m.version << ", expected " << ref.version;
        _consistency_throw(ss.str());
    }
    if (m.zone_nfreq != ref.zone_nfreq)
        _consistency_throw("mismatch in zone_nfreq");

    _check_double_vec_eq(m.zone_freq_edges, ref.zone_freq_edges, freq_eps, "zone_freq_edges");

    if (m.beamset != ref.beamset) {
        stringstream ss;
        ss << "mismatch in beamset: got " << m.beamset << ", expected " << ref.beamset;
        _consistency_throw(ss.str());
    }
    if (m.beam_ids != ref.beam_ids)
        _consistency_throw("mismatch in beam_ids");
    _check_double_vec_eq(m.beam_positions_x, ref.beam_positions_x, cosine_eps, "beam_positions_x");
    _check_double_vec_eq(m.beam_positions_y, ref.beam_positions_y, cosine_eps, "beam_positions_y");

    if (m.unix_ns_at_seq_0 != ref.unix_ns_at_seq_0) {
        stringstream ss;
        ss << "mismatch in unix_ns_at_seq_0: got " << m.unix_ns_at_seq_0
           << ", expected " << ref.unix_ns_at_seq_0;
        _consistency_throw(ss.str());
    }
    if (m.dt_ns_per_seq != ref.dt_ns_per_seq) {
        stringstream ss;
        ss << "mismatch in dt_ns_per_seq: got " << m.dt_ns_per_seq
           << ", expected " << ref.dt_ns_per_seq;
        _consistency_throw(ss.str());
    }
    if (m.seq_per_frb_time_sample != ref.seq_per_frb_time_sample) {
        stringstream ss;
        ss << "mismatch in seq_per_frb_time_sample: got " << m.seq_per_frb_time_sample
           << ", expected " << ref.seq_per_frb_time_sample;
        _consistency_throw(ss.str());
    }

    _check_double_eq(m.tel_origin_itrs_lat_deg, ref.tel_origin_itrs_lat_deg, angle_eps, "tel_origin_itrs_lat_deg");
    _check_double_eq(m.tel_origin_itrs_lon_deg, ref.tel_origin_itrs_lon_deg, angle_eps, "tel_origin_itrs_lon_deg");
    _check_double_vec_eq(m.tel_grid_x_axis,    ref.tel_grid_x_axis,    cosine_eps, "tel_grid_x_axis");
    _check_double_vec_eq(m.tel_grid_y_axis,    ref.tel_grid_y_axis,    cosine_eps, "tel_grid_y_axis");
    _check_double_vec_eq(m.tel_dish_elev_axis, ref.tel_dish_elev_axis, cosine_eps, "tel_dish_elev_axis");
    _check_double_vec_eq(m.tel_dish_vert_axis, ref.tel_dish_vert_axis, cosine_eps, "tel_dish_vert_axis");
    _check_double_eq(m.tel_dish_coelev_deg,    ref.tel_dish_coelev_deg, angle_eps,    "tel_dish_coelev_deg");
    _check_double_eq(m.tel_dish_separation_x_m, ref.tel_dish_separation_x_m, dish_sep_eps, "tel_dish_separation_x_m");
    _check_double_eq(m.tel_dish_separation_y_m, ref.tel_dish_separation_y_m, dish_sep_eps, "tel_dish_separation_y_m");

    _check_double_vec_eq(m.noise_variance, ref.noise_variance, variance_eps, "noise_variance");
}


// -------------------------------------------------------------------------------------------------
//
// Placeholder constants and factories for test / fixture use.


// Placeholder telescope and timekeeping values used by make_test_instance().
// Mirrors pirate_frb/run_server.py:_FAKE_XENGINE_PLACEHOLDERS so that C++ and
// Python test fixtures agree on a "reasonable" baseline. These values are
// chosen to pass validate() and check_sender_consistency() -- they do not
// represent any real telescope pointing.
static constexpr long _PH_unix_ns_at_seq_0 = 1772483060000000000L;
static constexpr long _PH_dt_ns_per_seq = 5120L;
static constexpr long _PH_seq_per_frb_time_sample = 256L;
static constexpr double _PH_tel_origin_itrs_lat_deg = 49.32075144444;
static constexpr double _PH_tel_origin_itrs_lon_deg = -119.62081125;
static constexpr std::array<double, 3> _PH_tel_grid_x_axis    {0.999974342398359362, -0.000037539331442772, -0.007163318767675494};
static constexpr std::array<double, 3> _PH_tel_grid_y_axis    {0.000065403387739210,  0.999992433220348809,  0.003889630373557614};
static constexpr std::array<double, 3> _PH_tel_dish_elev_axis {0.99999999838132391,  -0.000056897733584327,  0.0};
static constexpr std::array<double, 3> _PH_tel_dish_vert_axis {0.0, 0.0, 1.0};
static constexpr double _PH_tel_dish_coelev_deg = 0.0;
static constexpr double _PH_tel_dish_separation_x_m = 6.300156854906823;
static constexpr double _PH_tel_dish_separation_y_m = 8.500057809796308;


// Helper: fills beam_positions_{x,y} with a deterministic 2D grid spanning
// [-0.1, +0.1] in both coordinates. Beams are placed row-major into the
// smallest enclosing square grid (g x g where g = ceil(sqrt(nbeams))), with
// extreme positions at +/-0.1. Single-beam case maps to (0, 0).
static void _fill_test_beam_positions(std::vector<double> &bx, std::vector<double> &by, long nbeams)
{
    xassert(nbeams > 0);
    bx.assign(nbeams, 0.0);
    by.assign(nbeams, 0.0);

    long g = std::max(1L, long(std::ceil(std::sqrt(double(nbeams)))));
    if (g <= 1)
        return;  // single beam stays at origin

    double step = 0.2 / double(g - 1);
    for (long i = 0; i < nbeams; i++) {
        long row = i / g;
        long col = i % g;
        bx[i] = -0.1 + double(col) * step;
        by[i] = -0.1 + double(row) * step;
    }
}


// static member function
std::shared_ptr<XEngineMetadata>
XEngineMetadata::make_test_instance(const std::vector<long> &zone_nfreq_,
                                    const std::vector<double> &zone_freq_edges_,
                                    const std::vector<long> &beam_ids_)
{
    xassert(zone_nfreq_.size() > 0);
    xassert(beam_ids_.size() > 0);

    auto ret = std::make_shared<XEngineMetadata>();
    ret->version = 2;
    ret->zone_nfreq = zone_nfreq_;
    ret->zone_freq_edges = zone_freq_edges_;
    // freq_channels left empty (default).
    ret->beamset = 0;
    ret->beam_ids = beam_ids_;
    _fill_test_beam_positions(ret->beam_positions_x, ret->beam_positions_y, long(beam_ids_.size()));
    ret->noise_variance.assign(zone_nfreq_.size(), 1.0);

    ret->unix_ns_at_seq_0 = _PH_unix_ns_at_seq_0;
    ret->dt_ns_per_seq = _PH_dt_ns_per_seq;
    ret->seq_per_frb_time_sample = _PH_seq_per_frb_time_sample;

    ret->tel_origin_itrs_lat_deg = _PH_tel_origin_itrs_lat_deg;
    ret->tel_origin_itrs_lon_deg = _PH_tel_origin_itrs_lon_deg;
    ret->tel_grid_x_axis    = _PH_tel_grid_x_axis;
    ret->tel_grid_y_axis    = _PH_tel_grid_y_axis;
    ret->tel_dish_elev_axis = _PH_tel_dish_elev_axis;
    ret->tel_dish_vert_axis = _PH_tel_dish_vert_axis;
    ret->tel_dish_coelev_deg = _PH_tel_dish_coelev_deg;
    ret->tel_dish_separation_x_m = _PH_tel_dish_separation_x_m;
    ret->tel_dish_separation_y_m = _PH_tel_dish_separation_y_m;

    ret->validate();
    return ret;
}


// Helper: returns a random unit 3-vector (isotropically distributed).
// Rejection samples in the cube [-1, 1]^3 to avoid pole bias, then normalizes.
static std::array<double, 3> _rand_unit_vec(std::mt19937 &rng)
{
    while (true) {
        double x = ksgpu::rand_uniform(-1.0, 1.0, rng);
        double y = ksgpu::rand_uniform(-1.0, 1.0, rng);
        double z = ksgpu::rand_uniform(-1.0, 1.0, rng);
        double n2 = x*x + y*y + z*z;
        if ((n2 > 1.0e-6) && (n2 <= 1.0)) {
            double n = std::sqrt(n2);
            return { x/n, y/n, z/n };
        }
    }
}


// Helper: returns a random unit 3-vector orthogonal to 'a'.
// Generates a candidate random unit vector, subtracts its projection on 'a',
// and re-normalizes. Loops in the unlikely event of degeneracy.
static std::array<double, 3> _rand_orthogonal_unit_vec(const std::array<double, 3> &a, std::mt19937 &rng)
{
    while (true) {
        std::array<double, 3> v = _rand_unit_vec(rng);
        double dot = a[0]*v[0] + a[1]*v[1] + a[2]*v[2];
        v[0] -= dot * a[0];
        v[1] -= dot * a[1];
        v[2] -= dot * a[2];
        double n2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        if (n2 > 1.0e-6) {
            double n = std::sqrt(n2);
            return { v[0]/n, v[1]/n, v[2]/n };
        }
    }
}


// static member function
std::shared_ptr<XEngineMetadata> XEngineMetadata::make_random()
{
    using ksgpu::rand_int;
    using ksgpu::rand_uniform;

    std::mt19937 &rng = ksgpu::default_rng();

    auto ret = std::make_shared<XEngineMetadata>();
    ret->version = 2;

    // ---- Frequency zones ----

    long nzones = rand_int(1, 5, rng);
    long max_nfreq_per_zone = 100 / nzones;

    ret->zone_nfreq.resize(nzones);
    for (long i = 0; i < nzones; i++)
        ret->zone_nfreq[i] = rand_int(max_nfreq_per_zone/2, max_nfreq_per_zone+1, rng);

    // zone_freq_edges: nzones+1 sorted distinct values in [300, 1500] MHz.
    // freq_eps = 1e-3 in validate(); easily exceeded by uniform samples in this range.
    ret->zone_freq_edges.resize(nzones + 1);
    for (long i = 0; i < nzones + 1; i++)
        ret->zone_freq_edges[i] = rand_uniform(300.0, 1500.0, rng);
    std::sort(ret->zone_freq_edges.begin(), ret->zone_freq_edges.end());

    // freq_channels left empty (its content is only meaningful in receiver context).

    // ---- Beams ----

    ret->beamset = rand_int(0, 1024, rng);

    long nbeams = rand_int(1, 9, rng);

    // Distinct beam_ids drawn from [0, 1024). Reject-loop until all distinct.
    std::unordered_set<long> seen;
    ret->beam_ids.clear();
    while (long(ret->beam_ids.size()) < nbeams) {
        long id = rand_int(0, 1024, rng);
        if (seen.insert(id).second)
            ret->beam_ids.push_back(id);
    }

    // beam_positions in the unit disk: rejection sample.
    ret->beam_positions_x.resize(nbeams);
    ret->beam_positions_y.resize(nbeams);
    for (long i = 0; i < nbeams; i++) {
        while (true) {
            double bx = rand_uniform(-1.0, 1.0, rng);
            double by = rand_uniform(-1.0, 1.0, rng);
            if (bx*bx + by*by < 1.0) {
                ret->beam_positions_x[i] = bx;
                ret->beam_positions_y[i] = by;
                break;
            }
        }
    }

    // ---- Timekeeping ----

    ret->unix_ns_at_seq_0 = rand_int(long(1e18), long(2e18), rng);
    ret->dt_ns_per_seq = rand_int(1, long(1e6), rng);
    ret->seq_per_frb_time_sample = rand_int(1, long(1e4), rng);

    // ---- Telescope geometry ----

    ret->tel_origin_itrs_lat_deg = rand_uniform(-90.0, 90.0, rng);
    ret->tel_origin_itrs_lon_deg = rand_uniform(-180.0, 180.0, rng);

    ret->tel_grid_x_axis    = _rand_unit_vec(rng);
    ret->tel_grid_y_axis    = _rand_orthogonal_unit_vec(ret->tel_grid_x_axis, rng);
    ret->tel_dish_elev_axis = _rand_unit_vec(rng);
    ret->tel_dish_vert_axis = _rand_orthogonal_unit_vec(ret->tel_dish_elev_axis, rng);

    ret->tel_dish_coelev_deg = rand_uniform(0.0, 90.0, rng);
    ret->tel_dish_separation_x_m = rand_uniform(1.0, 20.0, rng);
    ret->tel_dish_separation_y_m = rand_uniform(1.0, 20.0, rng);

    // ---- Noise ----

    ret->noise_variance.resize(nzones);
    for (long i = 0; i < nzones; i++)
        ret->noise_variance[i] = rand_uniform(0.1, 10.0, rng);

    ret->validate();
    return ret;
}


}  // namespace pirate
