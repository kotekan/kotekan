#include "Telescope.hpp"

#include <mutex>              // for unique_lock
#include <shared_mutex>       // for shared_lock
#include <stdexcept>          // for invalid_argument
#include <algorithm>          // for copy, lower_bound, sort
#include <functional>         // for bind, _1, function
#include <limits>             // for numeric_limits

#include "configUpdater.hpp"  // for configUpdater
#include "geoUtil.hpp"        // for GeoFrame
#include "restServer.hpp"     // for restServer, connectionInstance
#include "timeUtil.hpp"       // for EOP, BareEOP, get_ERA_from_UT1, EOP_comp_time, eop_null
#include "fmt.hpp"            // for compile_string_to_view

using kotekan::connectionInstance;
using kotekan::restServer;

#define GIGA 1'000'000'000L

static constexpr double C = 2.99792458e8;
static constexpr double deg2rad = M_PI / 180.0;
static constexpr double arcsec2rad = M_PI / (180.0 * 3600);

static constexpr BareEOP dummy_bare_eop_first = {
    .t_inst_ns = 0, .delta_UT1_inst = 0.0, .xp_as = 0.0, .yp_as = 0.0};
static constexpr BareEOP dummy_bare_eop_last = {.t_inst_ns = std::numeric_limits<int64_t>::max(),
                                                .delta_UT1_inst = 0.0,
                                                .xp_as = 0.0,
                                                .yp_as = 0.0};

Telescope::Telescope(const std::string& tel_path, const std::string& log_level, bool require_eop,
                     const std::string& eop_updatable_config_path, const GeoFrame& frame) :
    _unique_name(tel_path),
    _frame(frame),
    _require_eop(require_eop) {
    set_log_level(log_level);
    set_log_prefix("/telescope");

    DEBUG("Building Telescope");

    // Initializing EOP table with dummy values
    _eop_table = {dummy_bare_eop_first.to_EOP(), dummy_bare_eop_last.to_EOP()};

    // Set up callbacks for updating EOP and sending time0_ns
    using namespace std::placeholders;

    // Subscribe to config updates if updatable_config is set.
    if (eop_updatable_config_path.length() > 0) {
        INFO("Subscribing {:s} to updatable config.", eop_updatable_config_path);
        kotekan::configUpdater::instance().subscribe(
            eop_updatable_config_path, std::bind(&Telescope::receive_eop_updates, this, _1));
    }

    INFO("Adding telescope REST GET endpoints");
    restServer& rest_server = restServer::instance();

    rest_server.register_get_callback(tel_path + "/time0_ns",
                                      std::bind(&Telescope::send_time0_ns, this, _1));
    rest_server.register_get_callback(tel_path + "/eop_table",
                                      std::bind(&Telescope::send_eop_table, this, _1));
}

Telescope::~Telescope() {
    // Must manually remove the GET callbacks
    restServer& rest_server = restServer::instance();
    rest_server.remove_get_callback(_unique_name + "/time0_ns");
    rest_server.remove_get_callback(_unique_name + "/eop_table");
    DEBUG("Removed REST GET endpoints");
}

const Telescope& Telescope::instance() {
    if (!tel_instance()) {
        FATAL_ERROR_NON_OO("Telescope singleton must be configured before use.");
    }

    return *tel_instance();
}

const Telescope& Telescope::instance(const kotekan::Config& config) {

    // This defaults to ICETelescope because the "Telescope" is virtual
    // and is not registered with the Factory.
    auto telescope_name = config.get_default<std::string>("/telescope", "name", "ICETelescope");
#if !defined(WITH_TESTS)
    if (telescope_name == "fake")
        WARN_NON_OO("To use the fake telescope, build with -DWITH_TESTS=ON");
#endif
    if (!FACTORY(Telescope)::exists(telescope_name)) {
        FATAL_ERROR_NON_OO("Requested telescope type {} is not registered", telescope_name);
    }

    tel_instance() = FACTORY(Telescope)::create_unique(telescope_name, config, "/telescope");

    return *tel_instance();
}

std::unique_ptr<Telescope>& Telescope::tel_instance() {
    // this must be declare in a function to ensure correct order of
    // desctructors when unwinding the ctor stack
    static std::unique_ptr<Telescope> the_tel_instance;

    return the_tel_instance;
}

freq_id_t Telescope::to_freq_id(stream_t stream) const {
    if (num_freq_per_stream() != 1) {
        throw std::invalid_argument(
            "Cannot use the to_freq_id(stream) call on a multi-frequency stream.");
    }
    return to_freq_id(stream, 0);
}

uint64_t Telescope::to_seq(int64_t time_ns) const {
    return to_seq(nanosec_i64_to_timespec(time_ns));
}

timespec Telescope::seq_length() const {
    auto dt_ns = seq_length_nsec();
    return {(time_t)(dt_ns / GIGA), (long)(dt_ns % GIGA)};
}

bool Telescope::receive_eop_updates(nlohmann::json& json) {
    try {
        // Fill a temporary table with the updated values.

        if (!json.contains("earth_orientation_parameter_table") && !_require_eop) {
            // If EOP is not required, say we succeeded and do nothing.
            WARN("EOP update did not contain `earth_orientation_parameter_table`, ignoring. EOP "
                 "table is unchanged.");
            return true;
        }
        std::vector<BareEOP> bare_eop_table = json.at("earth_orientation_parameter_table");

        size_t N = bare_eop_table.size();
        std::vector<EOP> tmp_eop_table(N);

        for (size_t i = 0; i < N; i++)
            tmp_eop_table.at(i) = bare_eop_table.at(i).to_EOP();

        if (tmp_eop_table.empty()) {
            // If table was empty, we're done here.

            if (_require_eop) {
                // If table was required. Report an error.
                ERROR("earth_orientation_parameter_table update contained no entries.",
                      _unique_name);
                return false;
            }

            // If table is not required, signal success and ignore the update.
            INFO("Ignoring `earth_orientation_parameter_table` update, it contained no entries. "
                 "EOP Table is unchanged.");
            return true;
        }

        // There's at least one entry in the table, sort it and replace the current table.

        // Sort chronologically
        std::sort(tmp_eop_table.begin(), tmp_eop_table.end(), EOP_comp_time);

        // Replace old table with new.
        {
            // Make sure no one is using the EOP table while we're updating it.
            std::unique_lock lock(_eop_lock);
            _eop_table = tmp_eop_table;
            INFO("Updated EOP Table with {:d} entries", _eop_table.size());
        }

    } catch (std::exception& e) {
        WARN("Telescope failed to read EOP update: {:s}", e.what());
        return false;
    }

    return true;
}

void Telescope::send_eop_table(connectionInstance& conn) {
    nlohmann::json reply;
    {
        std::shared_lock lock(_eop_lock);
        reply["eop_table"] = _eop_table;
    }
    conn.send_json_reply(reply);
}

void Telescope::send_time0_ns(connectionInstance& conn) {
    nlohmann::json reply;
    reply["time0_ns"] = to_time_ns(0);
    conn.send_json_reply(reply);
}

std::vector<EOP> Telescope::get_current_EOP_table() const {

    std::vector<EOP> tab_copy;
    {
        std::shared_lock lock(_eop_lock);
        tab_copy = _eop_table;
    }
    return tab_copy;
}

EOP Telescope::get_EOP_at_time(const timespec& ts_target) const {
    // extract the time in nanoseconds and then use the time_ns function.
    int64_t ts_target_ns = timespec_to_nanosec_i64(ts_target);
    return get_EOP_at_time_ns(ts_target_ns);
}

EOP Telescope::get_EOP_at_time_ns(int64_t t_target_ns) const {
    // Interpolate on the EOP table to find EOP for the given instrument time.
    EOP eop;

    eop.t_inst_ns = t_target_ns;

    // thread-safety: acquire lock on EOP table
    {
        std::shared_lock lock(_eop_lock);

        if (_eop_table.empty()) {
            WARN("EOP table is empty, cannot interpolate EOP at instrument time {:d} s + {:d} ns.",
                 t_target_ns / GIGA, t_target_ns % GIGA);
            return eop_null;
        }

        // _eop_table is always sorted by instrument time. Do a quick search
        // for the first table entry with larger time than the target.
        auto eop_b = std::lower_bound(_eop_table.begin(), _eop_table.end(), eop, EOP_comp_time);

        // DUT1, xp_as, and yp_as evolve slowly, on secular time scales, so we
        // interpolate these, and calculate ERA after.

        if (eop_b == _eop_table.begin()) {
            // Time is earlier than covered by the table, use the first value.
            eop.delta_UT1_inst = eop_b->delta_UT1_inst;
            eop.xp_as = eop_b->xp_as;
            eop.yp_as = eop_b->yp_as;
            if (t_target_ns < eop_b->t_inst_ns) {
                WARN("Requesting EOP earlier than in table. Requested time = {:d} s + {:d} ns. "
                     "Earliest "
                     "time = {:d} s + {:d} ns.",
                     t_target_ns / GIGA, t_target_ns % GIGA, eop_b->t_inst_ns / GIGA,
                     eop_b->t_inst_ns % GIGA);
            }
        } else if (eop_b == _eop_table.end()) {
            // Time is later than covered by the table, use the last value.
            auto eop_last = eop_b - 1;
            eop.delta_UT1_inst = eop_last->delta_UT1_inst;
            eop.xp_as = eop_last->xp_as;
            eop.yp_as = eop_last->yp_as;
            if (t_target_ns > eop_last->t_inst_ns) {
                WARN(
                    "Requesting EOP later than in table. Requested time = {:d} s + {:d} ns. Latest "
                    "time = "
                    "{:d} s + {:d} ns.",
                    t_target_ns / GIGA, t_target_ns % GIGA, eop_last->t_inst_ns / GIGA,
                    eop_last->t_inst_ns % GIGA);
            }
        } else {
            // Interpolate!
            auto eop_a = eop_b - 1;
            // t - ta in ns. Should be > 0
            int64_t diff_ns_a = t_target_ns - eop_a->t_inst_ns;
            // t - tb in ns. Should be < 0
            int64_t diff_ns_b = t_target_ns - eop_b->t_inst_ns;
            // tb - ta in ns.
            int64_t diff_ns = diff_ns_a - diff_ns_b;

            // weights for points a and b.
            double wb = diff_ns_a / ((double)diff_ns);
            double wa = 1.0 - wb;

            // interpolate
            eop.delta_UT1_inst = wa * eop_a->delta_UT1_inst + wb * eop_b->delta_UT1_inst;
            eop.xp_as = wa * eop_a->xp_as + wb * eop_b->xp_as;
            eop.yp_as = wa * eop_a->yp_as + wb * eop_b->yp_as;
        }
    } // eop_lock

    // now that we have a delta_UT1, can compute UT1 and ERA
    int64_t ut1 = get_UT1_from_time_ns(t_target_ns, eop.delta_UT1_inst);
    double era = get_ERA_from_UT1(ut1, nullptr);

    eop.t_ut1_ns = ut1;
    eop.ERA_deg = era;

    return eop;
}

EOP Telescope::get_EOP_at_UT1(int64_t t_ut1_ns) const {
    // Interpolate on the EOP table to find EOP for the given UT1 time.

    EOP eop;
    eop.t_ut1_ns = t_ut1_ns;

    // thread-safety: acquire lock on EOP table
    {
        std::shared_lock lock(_eop_lock);

        if (_eop_table.empty()) {
            WARN("EOP table is empty, cannot interpolate EOP at UT1 time {:d} s + {:d} ns.",
                 t_ut1_ns / GIGA, t_ut1_ns % GIGA);
            return eop_null;
        }

        // _eop_table is always sorted by instrument time. UT1 is monotonic
        // with instrument time, unless the Earth has been met with catastrophe.
        // Do a quick search for the first table entry with larger UT1 time than
        // the target.
        auto eop_b = std::lower_bound(_eop_table.begin(), _eop_table.end(), eop, EOP_comp_ut1);

        // DUT1, xp_as, and yp_as evolve slowly, on secular time scales, so we
        // interpolate these, and calculate ERA after.

        if (eop_b == _eop_table.begin()) {
            // Time is earlier than covered by the table, use the first value.
            eop.delta_UT1_inst = eop_b->delta_UT1_inst;
            eop.xp_as = eop_b->xp_as;
            eop.yp_as = eop_b->yp_as;
            WARN("Requesting EOP earlier than in table. Requested UT1 = {:d} s + {:d} ns. Earliest "
                 "UT1 "
                 "= {:d} s + {:d} ns.",
                 t_ut1_ns / GIGA, t_ut1_ns % GIGA, eop_b->t_ut1_ns / GIGA, eop_b->t_ut1_ns % GIGA);
        } else if (eop_b == _eop_table.end()) {
            // Time is later than covered by the table, use the last value.
            auto eop_last = eop_b - 1;
            eop.delta_UT1_inst = eop_last->delta_UT1_inst;
            eop.xp_as = eop_last->xp_as;
            eop.yp_as = eop_last->yp_as;
            WARN("Requesting EOP later than in table. Requested UT1 = {:d} s + {:d} ns. Latest UT1 "
                 "= "
                 "{:d} s + {:d} ns.",
                 t_ut1_ns / GIGA, t_ut1_ns % GIGA, eop_last->t_ut1_ns / GIGA,
                 eop_last->t_ut1_ns % GIGA);
        } else {
            // Interpolate! Target time = t, in table interval [ta, tb]
            auto eop_a = eop_b - 1;

            // t - ta in ns. Should be > 0
            int64_t diff_ns_a = t_ut1_ns - eop_a->t_ut1_ns;
            // t - tb in ns. Should be < 0
            int64_t diff_ns_b = t_ut1_ns - eop_b->t_ut1_ns;

            // tb - ta in ns.
            int64_t diff_ns = diff_ns_a - diff_ns_b;

            // weight for b point
            double wb = diff_ns_a / ((double)diff_ns);
            // weight for a point.
            double wa = 1.0 - wb;

            // interpolate.
            eop.delta_UT1_inst = wa * eop_a->delta_UT1_inst + wb * eop_b->delta_UT1_inst;
            eop.xp_as = wa * eop_a->xp_as + wb * eop_b->xp_as;
            eop.yp_as = wa * eop_a->yp_as + wb * eop_b->yp_as;
        }
    } // eop_lock

    // Now that we have a delta_UT1, can get t_inst and the ERA
    eop.t_inst_ns = get_time_ns_from_UT1(t_ut1_ns, eop.delta_UT1_inst);
    eop.ERA_deg = get_ERA_from_UT1(t_ut1_ns, nullptr);


    return eop;
}

double Telescope::get_itrs_lon_deg() const {
    return _frame.get_itrs_lon_deg();
}

double Telescope::get_itrs_lat_deg() const {
    return _frame.get_itrs_lat_deg();
}

double Telescope::get_ERAL_deg(EOP& eop) const {
    double era = eop.ERA_deg;
    double lon = _frame.get_itrs_lon_deg();
    double eral = era + lon; // Ignore TIO locator s', it is only ~12 micro arcseconds.

    // Reset eral to [0, 360)
    double n_off = std::floor(eral / 360.0);
    eral -= n_off * 360;

    return eral;
}

mat3x3d_t Telescope::get_grid_orientation() const {
    return _frame.get_R_topo_to_frame();
}

vec3d_t Telescope::vec_topo_to_grid(const vec3d_t& v_topo) const {
    return _frame.vec_topo_to_frame(v_topo);
}

vec3d_t Telescope::vec_topo_to_itrs(const vec3d_t& v_topo) const {
    return _frame.vec_topo_to_itrs(v_topo);
}

vec3d_t Telescope::vec_grid_to_topo(const vec3d_t& v_grid) const {
    return _frame.vec_frame_to_topo(v_grid);
}

vec3d_t Telescope::vec_itrs_to_topo(const vec3d_t& v_itrs) const {
    return _frame.vec_itrs_to_topo(v_itrs);
}

vec3d_t Telescope::vec_cirs_to_itrs(const vec3d_t& v_cirs, const EOP& eop) const {

    // IERS Conventions (2010) Chapter 5, Eq 5.1-5.3, and 5.5 give the
    // ITRS -> CIRS Transformation:
    //
    // [CIRS] = R(t) W(t) [ITRS]        Eq. 5.1
    //
    // W(t) = R3(s') R2(x') R1(y')      Eq. 5.3
    // R(t) = R3(-ERA)                  Eq. 5.5
    //
    // We ignore the s' contribution here, it's magnitude is only
    // microarcsecond.
    //
    // The inverse transformation reverses this, taking the negative of each
    // argument.
    double era = deg2rad * eop.ERA_deg;
    double xp = arcsec2rad * eop.xp_as;
    double yp = arcsec2rad * eop.yp_as;

    // 5.5 inverse
    vec3d_t v1 = vec3d_axes_rotation_R3(v_cirs, era);
    // 5.3, second factor, inverse
    vec3d_t v2 = vec3d_axes_rotation_R2(v1, -xp);
    // 5.3, first factor, inverse
    vec3d_t v_itrs = vec3d_axes_rotation_R1(v2, -yp);

    return v_itrs;
}

vec3d_t Telescope::vec_cirs_from_ra_dec(double ra_cirs_deg, double dec_cirs_deg) const {
    // Taking the ra & dec to be in CIRS frame
    double phi = deg2rad * ra_cirs_deg;
    double theta = deg2rad * (90 - dec_cirs_deg);
    
    // unit vector pointing to ra/dec in spherical coordinates
    // fixed to the Earth.  phi=0 ~ Greenwich
    vec3d_t n_cirs = {cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)};

    return n_cirs;
}

void Telescope::vec_cirs_to_ra_dec(const vec3d_t& v_cirs, double& ra_cirs_deg, double& dec_cirs_deg) const {

    ra_cirs_deg = atan2(v_cirs[1], v_cirs[0]) / deg2rad;
    dec_cirs_deg = 90.0 - acos(v_cirs[2]) / deg2rad;
}
    
vec3d_t Telescope::vec_grid_to_cirs(const vec3d_t& v_grid, const EOP& eop) const {
    vec3d_t v_topo = vec_grid_to_topo(v_grid);
    vec3d_t v_itrs = vec_topo_to_itrs(v_topo);
    vec3d_t v_cirs = vec_itrs_to_cirs(v_itrs, eop);

    return v_cirs;
}

vec3d_t Telescope::vec_cirs_to_grid(const vec3d_t& v_cirs, const EOP& eop) const {
    vec3d_t v_itrs = vec_cirs_to_itrs(v_cirs, eop);
    vec3d_t v_topo = vec_itrs_to_topo(v_itrs);
    vec3d_t v_grid = vec_topo_to_grid(v_topo);

    return v_grid;
}
    
vec3d_t Telescope::vec_cirs_ra_dec_to_grid(double ra_cirs_deg, double dec_cirs_deg, const EOP& eop) const {
    vec3d_t v_cirs = vec_cirs_from_ra_dec(ra_cirs_deg, dec_cirs_deg);
    return vec_cirs_to_grid(v_cirs, eop);
}

grid_idx_2d_t Telescope::element_index_to_grid_indices(uint64_t el_idx, ElementOrder ord) const {
    station_id_t st_id = element_index_to_station_id(el_idx, ord);
    return station_id_to_grid_indices(st_id);
}

vec3d_t Telescope::element_index_to_feed_position_m(uint64_t el_idx, ElementOrder ord) const {
    station_id_t st_id = element_index_to_station_id(el_idx, ord);
    return station_id_to_feed_position_m(st_id);
}

std::vector<grid_idx_2d_t> Telescope::get_grid_indices(uint64_t num_elements, ElementOrder ord) const {
    std::vector<grid_idx_2d_t> grid_indices(num_elements);

    for(uint64_t el_idx = 0; el_idx < num_elements; el_idx++) 
        grid_indices.at(el_idx) = element_index_to_grid_indices(el_idx, ord);

    return grid_indices;
}

std::vector<vec3d_t> Telescope::get_feed_positions_m(uint64_t num_elements, ElementOrder ord) const {
    std::vector<vec3d_t> feed_positions_m(num_elements);

    for(uint64_t el_idx = 0; el_idx < num_elements; el_idx++) 
        feed_positions_m.at(el_idx) = element_index_to_feed_position_m(el_idx, ord);

    return feed_positions_m;
}

void Telescope::fill_fringestop_phases_1d(double freq_MHz, const EOP& eop, const EOP& eop0,
                                          const std::vector<vec3d_t> feed_positions_m,
                                          std::vector<std::complex<float>>& phases) const {

    if (feed_positions_m.size() != phases.size()) 
        FATAL_ERROR("fill_fringestop: feed_positions_m size {:d} != phases size {:d}",
                     feed_positions_m.size(), phases.size());

    // Get the phase center (pointing vector) of the telescope.  Depends on the dish coelevation
    // angle, which is fixed during a run.
    vec3d_t n_grid0 = get_phase_center_in_grid_frame();

    // Take the pointing vector for the telescope and find it in the CIRS frame at ERA0.
    // This is the point we are attempting to stop the fringes at.
    vec3d_t n_cirs = vec_grid_to_cirs(n_grid0, eop0);

    // Now, given this CIRS vector, find its components in the telescope
    // frame at the requested (current) ERA
    vec3d_t n_grid = vec_cirs_to_grid(n_cirs, eop);

    // n_grid is now (at ERA) the point on the sky which will be at the
    // phase center (n_grid0) at ERA0.

    // wavenumber for this frequency
    double k = 2 * M_PI * 1e6 * freq_MHz / C;

    for (uint64_t i = 0; i < feed_positions_m.size(); i++) {
        double phase = -k
                       * (feed_positions_m[i][0] * (n_grid[0] - n_grid0[0])
                          + feed_positions_m[i][1] * (n_grid[1] - n_grid0[1])
                          + feed_positions_m[i][2] * (n_grid[2] - n_grid0[2]));

        phases[i] = {static_cast<float>(cos(phase)), static_cast<float>(sin(phase))};
    }
}

vec3d_t Telescope::vec_itrs_to_cirs(const vec3d_t& v_itrs, const EOP& eop) const {

    // IERS Conventions (2010) Chapter 5, Eq 5.1-5.3, and 5.5 give the
    // ITRS -> CIRS Transformation:
    //
    // [CIRS] = R(t) W(t) [ITRS]        Eq. 5.1
    //
    // W(t) = R3(s') R2(x') R1(y')      Eq. 5.3
    // R(t) = R3(-ERA)                  Eq. 5.5
    //
    // We ignore the s' contribution here, it's magnitude is only
    // microarcsecond.

    double era = deg2rad * eop.ERA_deg;
    double xp = arcsec2rad * eop.xp_as;
    double yp = arcsec2rad * eop.yp_as;

    // 5.3 (First factor in W)
    vec3d_t v1 = vec3d_axes_rotation_R1(v_itrs, yp);
    // 5.3 (Second factor in W)
    vec3d_t v2 = vec3d_axes_rotation_R2(v1, xp);
    // 5.5
    vec3d_t v_cirs = vec3d_axes_rotation_R3(v2, -era);

    return v_cirs;
}

std::string ElementOrder_to_string(const ElementOrder& o) {
    switch(o) {
        case ElementOrder::CHIMECorrelator:
            return "CHIMECorrelator";
        case ElementOrder::CHIMECylinder:
            return "CHIMECylinder";
        case ElementOrder::CHIMEBeamformer:
            return "CHIMEBeamformer";
        case ElementOrder::CHORDEarly:
            return "CHORDEarly";
        case ElementOrder::CHORDBeamformer:
            return "CHORDBeamformer";
        default:
            FATAL_ERROR_NON_OO("Unknown ElementOrder: {:d}", static_cast<int32_t>(o));
    }
}

ElementOrder ElementOrder_from_string(const std::string& s) {
    if (s == "CHIMECorrelator")
        return ElementOrder::CHIMECorrelator;
    else if (s == "CHIMECylinder")
        return ElementOrder::CHIMECylinder;
    else if (s == "CHIMEBeamformer")
        return ElementOrder::CHIMEBeamformer;
    else if (s == "CHORDEarly")
        return ElementOrder::CHORDEarly;
    else if (s == "CHORDBeamformer")
        return ElementOrder::CHORDBeamformer;

    FATAL_ERROR_NON_OO("Unknown ElementOrder: {:s}", s);
}

std::ostream& operator<<(std::ostream& os, const ElementOrder& o) {
    os << ElementOrder_to_string(o);
    return os;
}

std::string format_as(const ElementOrder& o) {
    return ElementOrder_to_string(o);
}

void to_json(nlohmann::json& j, const ElementOrder& o) {
    j = ElementOrder_to_string(o);
}

void from_json(const nlohmann::json& j, ElementOrder& o) {
    o = ElementOrder_from_string(j);
}
