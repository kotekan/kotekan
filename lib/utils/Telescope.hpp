#ifndef TELESCOPE_HPP
#define TELESCOPE_HPP

#include <time.h>              // for timespec, size_t
#include <json.hpp>            // for json
#include <exception>           // for exception
#include <memory>              // for unique_ptr
#include <shared_mutex>        // for shared_mutex
#include <string>              // for string, basic_string
#include <cstdint>             // for uint64_t, int64_t, uint32_t, uint8_t, UINT32_MAX
#include <vector>              // for vector

#include "Config.hpp"          // for Config
#include "factory.hpp"         // for FACTORY, CREATE_FACTORY, REGISTER_NAMED_TYPE_WITH_FACTORY
#include "geoUtil.hpp"         // for GeoFrame
#include "kotekanLogging.hpp"  // for ERROR, kotekanLogging
#include "restServer.hpp"      // for connectionInstance
#include "timeUtil.hpp"        // for EOP
#include "fmt.hpp"             // for compile_string_to_view, format

// Create the abstract factory for generating patterns
class Telescope;

CREATE_FACTORY(Telescope, const kotekan::Config&, const std::string&);
#define REGISTER_TELESCOPE(TelescopeType, name)                                                    \
    REGISTER_NAMED_TYPE_WITH_FACTORY(Telescope, TelescopeType, name)


using nyquist_zone_t = std::uint8_t;
using freq_id_t = std::uint32_t; // logical ID, not necessarily an index
#define FREQ_ID_NOT_SET UINT32_MAX
using station_id_t = std::uint32_t;
using grid_idx_2d_t = std::array<int64_t, 2>;  // signed, use -1 as a sentinel for an unfilled
                                               // location.

/**
 * @brief Enum labelling the ordering of feed elements in memory.
 */
enum class ElementOrder : int32_t {
    // Order as received by the CHIME X-engine. Slightly scrambled, defined via table.
    CHIMECorrelator = 0,
    
    // Arrays are [Cylinder][P][Feed] where Cylinder counts cylinders west to east, P counts polarizations,
    // Feed counts feeds in a cylinder south to north.
    CHIMECylinder = 1,

    // Arrays are [P][D], P = polarizations, D = feed, ordered as in Cylinder
    CHIMEBeamformer = 2,        

    // Ordering for pre-pathfinder CHORD. Arrays are [D][P], D-ordering is defined via table.
    CHORDEarly = 3,   
    
    // Ordering for production CHORD. Arrays are [P][D], D-ordering is defined via table
    CHORDBeamformer = 4,  
};


std::string ElementOrder_to_string(const ElementOrder& o);   /// Convert an ElementOrder to a string
ElementOrder ElementOrder_from_string(const std::string& s); /// Convert a string to an ElementOrder
std::ostream& operator<<(std::ostream& os, const ElementOrder& o); 
std::string format_as(const ElementOrder& o);
void to_json(nlohmann::json& j, const ElementOrder& o);
void from_json(const nlohmann::json& j, ElementOrder& o);


/**
 * @brief A type for the stream ID.
 *
 * This is the external interface for it and *must* be used instead of directly
 * accessing the chimeMetadata::stream_ID member.
 **/
struct stream_t {
    uint64_t id;
};

/**
 * @brief A class to hold telescope specific functionality.
 *
 * This serves as a generic Telescope base class. It cannot be instantiated,
 * only derived classes can.  It provides virtual hooks for routines to map
 * between sequence number and instrument time, for returning sampling
 * parameters (raw sample rate, frequency spacing, map from freq_id to physical
 * frequency, etc), and for associating feed elements in memory with physical locations.
 * It contains the Earth Orientation Parameter (EOP) table,
 * which holds data to compute UT1 time and CIRS coordinate transformations from
 * instrument time. It facilitates transformations between feed baselines to CIRS coordinates and back.
 *
 * Coordinate frames used by the Telescope
 * ---------------------------------------
 *
 * ## GRID ##
 *  The GRID frame is an orthonormal Cartesian coordinate system whose:
 *      - x-axis is parallel to fiducial East/West feed separation vector (pointing ~East).  
 *      - y-axis is parallel to fiducial North/South feed separation vector (pointing ~North).  
 *      - z-axis is normal to the fiducial feed plane (pointing ~Up).
 *
 *  The actual telescope feed grid on the ground is not perfectly square.  The GRID frame is an orthonormal
 *  frame which fits the physical feed locations as well as possible, as determined by the telescope
 *  developers.
 *
 *  Feed Positions are returned in the GRID frame.
 *
 *  The GRID origin is an optional `offset` from the TOPO origin
 *
 * ## TOPO ##
 *  The Topocentric (TOPO) frame is an orthonormal Cartesian coordinate system located at a position on the
 *  Earth.
 *      - x-axis is parallel to local geodetic East (increasing longitude)
 *      - y-axis is parallel to local geodecic North (increasing latitude)
 *      - z-axis is parallel to local geodetic Up (increasing altitude)
 *  The TOPO origin is at a given Latitude and Longitude in ITRS coordinates.
 *
 * ## ITRS ##
 *  The International Terrestrial Reference System (ITRS) is the internationally agreed upon coordinate
 *  system for the planet Earth. This is the coordinate system where geodetic latitude and longitude live.
 *  The IAU and IERS define the transformations between ITRS and astronomical coordinates.
 *  ITRS also has a Cartesian representation with:
 *      - x-axis directed through latitude = 0 (the Equator), longitude = 0
 *      - y-axis directed through latitude = 0 (the Equator), longitude = 90 degrees East
 *      - z-axis directed through latitude = 90 (the North Pole)
 *  The ITRS origin is located at Earth Barycenter.
 *
 * ## CIRS ##
 *  The Celestial Intermediate Reference System (CIRS) is a set of celestial coordinates defined by the IAU
 *  and IERS. This system is non-rotating with respect to the distant stars, and is aligned with the Earth's
 *  instantaneous axis of rotation. It is an intermediate step between ITRS (Earth-based) and ICRS (the fixed
 *  stars). In Kotekan we do not need realtime knowledge of the ICRS so we only go as far as the CIRS, which
 *  encodes the full rotational state of the Earth but not the precession/nutation of its axis.
 *
 *  We track targets in the sky by following their fixed CIRS coordinates.
 *
 *  The ITRS <-> CIRS transformation is time dependent, and encoded by the time-dependent Earth Orientation
 *  Parameters (EOP). 
 *
 * The Main Array Grid
 * -------------------
 *
 *  Kotekan Telescopes are formed of a "main array" of feeds in a rectilinear grid on the Earth. This grid
 *  must be planar (flat) to a good approximation, but otherwise may have arbitrary orientation on the ground.
 *
 *  We assume (for now) the grid is near-perfect rectilinear, with feeds evenly spaced along orthogonal axes
 *  X and Y. X is the easterly-directed separation spacing, and Y is the northerly-directed spacing. Feeds
 *  in this grid have a 2D `grid_index`: (`grid_idx_x`, `grid_idx_y`).  These count from 0 beginning in the
 *  southwest corner of the main array, so any feed in the main array has NON-NEGATIVE grid indices.
 *
 *  Some feeds may not be in the main array (RFI Antennae, external telescopes, maser feeds, etc). These feeds
 *  have `grid_index` = (-1, -1).
 *
 *  Feeds may not be exactly on station. Their 3D position returned by `get_feed_positions_m()` may include
 *  per-feed displacements from the fiducial station position.  Non-main-array feeds may also have a feed
 *  position. Feed positions are given in the GRID frame.
 *
 * ElementOrder
 * ------------
 *
 *  Different telescopes (and different parts of the pipeline for the same telescope) may place their feeds
 *  in memory via different arrangements. To return the feed position corresponding to a particular element
 *  in a data array, you must provide the element index AND an ElementOrder variable specifying the ordering
 *  of this array. Different orders may, for instance, transpose polarization and dish, or may simply permute
 *  ordering of dishes within the dish axis.
 *  
 *
 * REST Endpoints
 *
 * @endpoint    /time0_ns   GET     Returns a JSON object with a single field
 *                                  "time0_ns" which contains the instrument time
 *                                  in nanoseconds at fpga_seq_num = 0. Instrument
 *                                  time at seq = 0 is represented as a UNIX time
 *                                  with nanosecond resolution.
 * @endpoint    /eop_table  GET     Returns a JSON object with a single field "eop_table" which
 *                                  contains a list of EOP objects. Each EOP object contains 6
 *                                  fields:
 *                                  - t_inst_ns            int64   Instrument time in nanoseconds.
 *                                  - t_ut1_ns             int64   UT1 time in nanoseconds since
 *                                                              2451545.0 JD(UT1).
 *                                  - delta_UT1_inst    double  Difference in seconds between
 *                                                              UT1 and Instrument time.
 *                                  - ERA_deg           double  Earth Rotation Angle in degrees.
 *                                  - xp_as             double  Polar Motion x', arcseconds.
 *                                  - yp_as             double  Polar Motion y', arcseconds.
 *
 * Updatable Config
 *
 * @conf    eop_updatable_config    Optional. If not present, EOP table will be a dummy and only
 *                                  null values (all 0s) will be returned by get_EOP functions.
 *                                  If present, the config path to an updatable config field
 *                                  containing the field "earth_orientation_parameter_table",
 *                                  which is a list of BareEOP objects. Each contains 4
 *                                  fields:
 *                                  - t_inst_ns      int     Instrument time in nanoseconds.
 *                                  - delta_UT1_inst    double  As in EOP.
 *                                  - xp_as             double  As in EOP.
 *                                  - yp_as             double  As in EOP.
 *                                  Upon receiving an update, the entire EOP table is replaced
 *                                  with the new table. UT1 and ERA values are calculated from
 *                                  the given t_inst_ns and delta_UT1_inst.
 *
 * To maintain continuity, tools updating the EOP table should first GET the current table, and
 * only update or add values at least two entries in the future.  The table is linearly
 * interpolated between the most recent past & future entries, so changing the closest future
 * entry will immediately change values currently being used in Kotekan.
 *
 * For example, given a current table:
 * [Jan 1 UTC 00:00:00,
 *  Jan 2 UTC 00:00:00,
 *  Jan 3 UTC 00:00:00,
 *  Jan 4 UTC 00:00:00]
 *
 *  An update on Jan 2 UTC 12:00:00 should keep the exact Jan 2 and Jan 3 entries as they are
 *  currently being used for interpolation.  Jan 4 may be modified, Jan 5 (or later) could be
 *  added, and Jan 1 can be removed.
 **/
class Telescope : public kotekan::kotekanLogging {

public:
    /**
     * @brief Construct the telescope singleton.
     *
     * @param  config  Kotekan configuration.
     *
     * @returns        A reference to the singleton instance.
     **/
    static const Telescope& instance(const kotekan::Config&);

    /**
     * @brief Get a reference to the singleton Telescope instance.
     *
     * @returns   The telescope instance.
     **/
    static const Telescope& instance();


    virtual ~Telescope();

    /**
     * @brief   Get the type name of this telescope object.
     *
     * @returns     The type of the telescope object as a string.
     **/
    std::string get_name() const {
        return FACTORY(Telescope)::label(*this);
    }

    /**
     * @brief   Cast this telescope object to a specific type.
     *
     * @returns     const reference of the specified Telesope type
     *
     * @throws      Exception if the cast is invalid. Can happen if the
     *              kotekan config is initializing an incompatible
     *              type of Telescope
     **/
    template<typename T>
    const T& cast() const {
        try {
            return dynamic_cast<const T&>(*this);
        } catch (const std::exception& e) {
            ERROR("Could not cast Telescope of type {:s} to type {:s}", this->get_name(),
                  FACTORY(Telescope)::label<T>());
            throw;
        }
    }

    /**
     * Get the frequency ID from the FPGA stream ID.
     *
     * @param  stream  The generic stream ID.
     *
     * @returns        The integer frequency ID.
     **/
    virtual freq_id_t to_freq_id(stream_t stream) const;


    /**
     * Get the frequency ID from the FPGA stream ID.
     *
     * @param  stream  The generic stream ID.
     * @param  ind     The index for multifrequency streams.
     *
     * @returns        The integer frequency ID.
     **/
    virtual freq_id_t to_freq_id(stream_t stream, uint32_t ind) const = 0;


    /**
     * Get the physical frequency in MHz of the specified freq ID.
     *
     * @param  freq_id  The frequency ID.
     *
     * @returns         The central frequency in MHz.
     **/
    virtual double to_freq_MHz(freq_id_t freq_id) const = 0;


    /**
     * Get the physical frequency in MHz of the specified channel.
     *
     * The baseclass implementation just calls
     * `to_freq_MHz(to_freq_id(args))`, override with a custom implementation
     * to save a function call.
     *
     * @param  args  Any arguments accepted by `to_freq_id`.
     *
     * @returns      The central frequency in MHz.
     **/
    template<typename... Args>
    double to_freq_MHz(Args... args) const {
        return to_freq_MHz(to_freq_id(args...));
    }


    /**
     * @brief Get the number of frequencies per stream.
     *
     * @return  The number of frequencies on a stream.
     **/
    virtual size_t num_freq_per_stream() const = 0;


    /**
     * @brief Get the total number of frequencies channels.
     *
     * This is the upper bound for freq_id.
     *
     * @return  The total number of frequency channels.
     **/
    virtual size_t num_freq() const = 0;

    /**
     * @brief Get the frequency width of a given channel.
     *
     * @return  The width of the frequency channel in MHz.
     **/
    virtual double freq_width_MHz(freq_id_t freq_id) const = 0;

    /**
     * @brief Get which Nyquist zone we are in.
     *
     * @return  The Nyquist zone.
     **/
    virtual nyquist_zone_t nyquist_zone() const = 0;

    /**
     * Convert a sequence number into an Instrument (~UNIX epoch) time.
     *
     * @param  seq  The sequence number.
     *
     * @return  The corresponding UNIX time.
     **/
    virtual timespec to_time(uint64_t seq) const = 0;

    /**
     * @brief   Return the Instrument time in ns corresponding to the given fpga sequence number as
     * an int64_t.
     */
    virtual int64_t to_time_ns(uint64_t seq) const = 0;

    /**
     * @brief Convert an Instrument (~UNIX epoch) time into the nearest sequence number.
     *
     * @note When there is not an exact correspondence between the given time
     *       and FPGA sequence numbers, this routine will return the latest valid
     *       FPGA sequence number before the given timestamp.
     *
     * @param  time  The Instrument time (a UNIX time unless a leap second has occured since
     *          instrument startup, or one will occur in the next 24 hours).
     *
     * @return  The corresponding sequence number.
     **/
    virtual uint64_t to_seq(timespec time) const = 0;

    /**
     * @brief Convert an Instrument (~UNIX epoch) time into the nearest sequence number.
     *
     * @note When there is not an exact correspondence between the given time
     *       and FPGA sequence numbers, this routine will return the latest valid
     *       FPGA sequence number before the given timestamp.
     *
     * @param  time_ns  The Instrument time (a UNIX time unless a leap second has occured since
     *              instrument startup, or one will occur in the next 24 hours) in nanoseconds.
     *
     * @return  The corresponding sequence number.
     **/
    virtual uint64_t to_seq(int64_t time_ns) const;


    /**
     * @brief Is a precise time source available?
     *
     * @return  True if the GPS time is available.
     **/
    virtual bool gps_time_enabled() const = 0;


    /**
     * @brief Get the length of an FPGA sequence number tick.
     *
     * @return  Length of an FPGA sequence number tick.
     **/
    virtual timespec seq_length() const;

    /**
     * @brief Get the length of an FPGA sequence number tick.
     *
     * @return  Length of an FPGA sequence number tick.
     **/
    virtual uint64_t seq_length_nsec() const = 0;

    /**
     * @brief Convert an element array index in a particular ordering into the corresponding station_id
     *
     * @param el_idx    Index into the element axis of an array
     * @param ord       The ordering of the array
     *
     * @return  Station ID for this index.
     **/
    virtual station_id_t element_index_to_station_id(uint64_t el_idx, ElementOrder ord) const = 0;

    /**
     * @brief Convert a Station ID into an element array index in a particular ordering.
     *
     * @param st_id Station ID of an input element.
     * @param ord   The ordering for the output index.
     *
     * @return  element array index for this Station ID.
     **/
    virtual uint64_t station_id_to_element_index(station_id_t st_id, ElementOrder ord) const = 0;

    /**
     * @brief Return the integral grid location for this station ID if it is an input in the 
     * main array, otherwise return (-1, -1).
     *
     * @param st_id Station ID of an input element.
     *
     * @return  Grid coordinates (idx_x, idx_y) for this Station ID, or (-1, -1) if this station
     * is not in the main array.
     **/
    virtual grid_idx_2d_t station_id_to_main_array_grid_indices(station_id_t st_id) const = 0;

    /**
     * @brief Return the 3D position in the GRID frame of this station ID.
     
     * @param st_id Station ID of an input element.
     *
     * @return 3D coordinates in meters for this station ID in the GRID frame.
     **/
    virtual vec3d_t station_id_to_feed_position_m(station_id_t st_id) const = 0;

    /**
     * @brief   Return the feed separation in the Grid-X direction in meters.
     **/
    virtual double get_feed_separation_x_m() const = 0;

    /**
     * @brief   Return the feed separation in the Grid-Y direction in meters.
     **/
    virtual double get_feed_separation_y_m() const = 0;

    /**
     * @brief   Return the size of the main array grid in the X (~East) direction. e.g.
     *          a 2 x 3-dish array would return 2 (assuming 2 dishes E/W and 3 dishes N/S).
     *          grid_idx[0] must be less than this.
     **/
    virtual uint64_t get_grid_size_x() const = 0;

    /**
     * @brief   Return the size of the main array grid in the Y (~North) direction. e.g.
     *          a 2 x 3-dish array would return 3 (assuming 2 dishes E/W and 3 dishes N/S).
     *          grid_idx[1] must be less than this.
     **/
    virtual uint64_t get_grid_size_y() const = 0;

    /**
     * @brief   Return the outward-directed 3D pointing vector for the telescope phase center in the GRID
     *          frame: n[3] = {nx, ny, nz},  |n| = 1.0.
     * For CHIME this should be simply the zenith (n ~ {0, 0, 1}),
     * for CHORD this is the boresight of the dishes, and depends on their co-elevation.
     **/
    virtual vec3d_t get_phase_center_in_grid_frame() const = 0;

    /**
     * @brief   Return a copy of the current EOP table.
     **/
    std::vector<EOP> get_current_EOP_table() const;

    /**
     * @brief   Return the EOP at the desired instrument time. Will interpolate
     *          over table, use first or last entry if target time is out of
     *          table range.
     *
     * @param   ts  Target instrument time, as a timespec.
     **/
    EOP get_EOP_at_time(const timespec& ts) const;

    /**
     * @brief   Return the EOP at the desired instrument time. Will interpolate
     *          over table, use first or last entry if target time is out of
     *          table range.
     *
     * @param   t_ns  Target instrument time in nanoseconds.
     **/
    EOP get_EOP_at_time_ns(int64_t t_ns) const;

    /**
     * @brief   Return the EOP at the desired UT1 time. Will interpolate
     *          over table, using the first or last entry if target time is
     *          out of table range.
     *
     * @param   ts  Target UT1 time, in nanoseconds since J2000(UT1) int64_t
     **/
    EOP get_EOP_at_UT1(int64_t ut1) const;

    /**
     * @brief   Compute the local ERA (eral in SOFA) at the telescope site.
     *
     *  The local ERA is:
     *
     *      ERAL = ERA + longitude(ITRS) + s',
     *
     *  where:
     *      ERA is the Earth Rotation Angle (era00 in SOFA),
     *      longitude(ITRS) is the geodetic longitude of the site in ITRS
     *      s' is the TIO locator (sp00 in SOFA)
     *
     *  This is the equivalent to Local Apparent Sidereal Time (LAST) in the CIO-based
     *  coordinate systems implemented by the IAU in 2000.
     *
     *  The s' is very small, it accrues at 47 microarcseconds per century, and is
     *  ignored in this calculation.
     *
     * @param   eop  An EOP object for the time at which the ERAL is requested.
     *
     * @return The local ERA in degrees.
     **/
    double get_ERAL_deg(EOP& eop) const;

    /**
     * @brief   Return the longitude of the instrument.
     **/
    double get_itrs_lon_deg() const;

    /**
     * @brief   Return the latitude of the instrument.
     **/
    double get_itrs_lat_deg() const;

    /**
     * @brief   Return the Topo -> Grid frame rotation matrix.
     **/
    mat3x3d_t get_grid_orientation() const;

    /**
     * @brief   Transform the given vector from topocentric to grid coords.
     *
     * @param   v_topo  Vector in topocentric coordinates.
     **/
    vec3d_t vec_topo_to_grid(const vec3d_t& v_topo) const;

    /**
     * @brief   Transform the given vector from grid to topocentric coords.
     *
     * @param   v_grid  Vector in grid coordinates.
     **/
    vec3d_t vec_grid_to_topo(const vec3d_t& v_grid) const;

    /**
     * @brief   Transform the given vector from ITRS to topocentric coords.
     *
     * @param   v_topo  Vector in ITRS coordinates.
     **/
    vec3d_t vec_itrs_to_topo(const vec3d_t& v_itrs) const;

    /**
     * @brief   Transform the given vector from topocentric to ITRS coords.
     *
     * @param   v_topo  Vector in topocentric coordinates.
     **/
    vec3d_t vec_topo_to_itrs(const vec3d_t& v_topo) const;

    /**
     * @brief   Transform the given vector from CIRS to ITRS coords.
     *
     * @param   v_topo  Vector in CIRS coordinates.
     * @param   eop     EOP for time of transformation.
     **/
    vec3d_t vec_cirs_to_itrs(const vec3d_t& v_cirs, const EOP& eop) const;

    /**
     * @brief   Transform the given vector from ITRS to CIRS coords.
     *
     * @param   v_topo  Vector in ITRS coordinates.
     * @param   eop     EOP for time of transformation.
     **/
    vec3d_t vec_itrs_to_cirs(const vec3d_t& v_itrs, const EOP& eop) const;

    /**
     * @brief   Return an observing vector (normalized vec3) in CIRS
     *          coordinates, corresponding to the given CIRS RA and DEC.
     * @param   ra_cirs_deg  Target Right Ascension in CIRS frame in degrees
     * @param   dec_cirs_deg Target Declination in CIRS frame in degrees
     **/
    vec3d_t vec_cirs_from_ra_dec(double ra_cirs_deg, double dec_cirs_deg) const;

    /**
     * @brief   Return the CIRS RA & DEC corresponding to the given observing vector
     *          (normalized vec3) in CIRS coordinates.
     * @param   v_cirs          Input vector in CIRS coordinates.
     * @param   ra_cirs_deg     Reference to return target Right Ascension in CIRS frame in degrees
     * @param   dec_cirs_deg    Reference to return target Declination in CIRS frame in degrees
     **/
    void vec_cirs_to_ra_dec(const vec3d_t& v_cirs, double& ra_cirs_deg, double& dec_cirs_deg) const;
    
    /**
     * @brief   Transform the given vector from GRID to CIRS coords.
     *
     * @param   v_grid  Vector in GRID coordinates.
     * @param   eop     EOP for time of transformation.
     **/
    vec3d_t vec_grid_to_cirs(const vec3d_t& v_grid, const EOP& eop) const;

    /**
     * @brief   Transform the given vector from CIRS to GRID coords.
     *
     * @param   v_cirs  Vector in CIRS coordinates.
     * @param   eop     EOP for time of transformation.
     **/
    vec3d_t vec_cirs_to_grid(const vec3d_t& v_cirs, const EOP& eop) const;
    
    /**
     * @brief   Return an observing vector (normalized vec3) in GRID
     *          coordinates, corresponding to the given CIRS RA and DEC.
     * @param   ra  Target Right Ascension in CIRS frame.
     * @param   dec Target Declination in CIRS frame.
     * @param   eop EOP for the time of observation.
     **/
    vec3d_t vec_cirs_ra_dec_to_grid(double ra_cirs_deg, double dec_cirs_deg, const EOP& eop) const;

    grid_idx_2d_t element_index_to_main_array_grid_indices(uint64_t el_idx, ElementOrder ord) const;
    vec3d_t element_index_to_feed_position_m(uint64_t el_idx, ElementOrder ord) const;

    std::vector<grid_idx_2d_t> get_main_array_grid_indices(uint64_t num_elements, ElementOrder ord) const;
    std::vector<vec3d_t> get_feed_positions_m(uint64_t num_elements, ElementOrder ord) const;

    /**
     * @brief   Compute the fringestopping phases for the given feed locations.
     *
     * @param   freq_MHz Frequency to compute phases for.
     * @param   eop     Current EOP.
     * @param   eop0    EOP of phase reference time. if eop=eop0 all phases are
     *                  1.0
     * @param   feed_posisions_m    The 3D feed positions in the telescope grid frame in meters. The
     *                  positions returned from `station_id_to_feed_position_m`.
     * @param   phases  Vector of std::complex<double>, with size equal to feed_positions_m. The
     *                  phase for each position will be written to this vector.
     **/
    void fill_fringestop_phases_1d(double freq_MHz, const EOP& eop, const EOP& eop0,
                                              const std::vector<vec3d_t> feed_positions_m,
                                              std::vector<std::complex<float>>& phases) const;


private:
    static std::unique_ptr<Telescope>& tel_instance();

protected:
    /**
     * @brief   The primary constructor which should be called by derived classes
     *          upon instantiation.
     *
     * This constructor sets up the logging and REST endpoints for Earth
     * Orientation Parameters (EOP) and time0. Implement a specific constructor
     * in a derived class to parse the config, and call this one to make sure
     * the logging is done correctly and endpoints are active.
     *
     * @param   tel_path    Path to the telescope in the Config (e.g. /telescope)
     * @param   log_level   The level to set logging at.
     * @param   require_eop Whether to require a valid EOP table.
     * @param   eop_updatable_config_path   The value of "eop_updatable_config" in
     *          the telescope Config, pointing to the updatable field which
     *          contains "earth_orientation_parameter_table"
     * @param   frame       Object containing the position of the telescope on the Earth
     *                      and the orientation of the feed grid axes.
     **/
    Telescope(const std::string& tel_path, const std::string& log_level, bool require_eop,
              const std::string& eop_updatable_config_path, const GeoFrame& frame);

    /**
     * @brief Callback to update EOP data
     *
     * @param json JSON reference of the config
     */
    bool receive_eop_updates(nlohmann::json& json);

    /**
     * @brief   Callback to send current EOP table
     *
     * @param   conn    Kotekan connection.
     */
    void send_eop_table(kotekan::connectionInstance& conn);

    /**
     * @brief   Callback to send time0_ns value
     *
     * @param   conn    Kotekan connection.
     */
    void send_time0_ns(kotekan::connectionInstance& conn);

    /**
     * The telescope's name in the config
     */
    const std::string _unique_name;

    /**
     *  Stores information about the geographic position and orientation of the telescope
     *  on the Earth.
     */
    const GeoFrame _frame;

    /**
     * Whether to require an EOP table. If false and no (or an empty) EOP table
     * is provided, the telescope will return a 0 EOP when queried.
     */
    const bool _require_eop;

    /**
     * This is the Earth Orientation Parameter (EOP) table used to determine
     * UT1 time and Earth Rotation Angle.
     */
    std::vector<EOP> _eop_table;

    /**
     * This mutex locks access to the _eop_table which can be updated via REST
     * calls.
     */
    mutable std::shared_mutex _eop_lock;
};

#endif // TELESCOPE_HPP
