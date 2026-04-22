#ifndef TELESCOPE_HPP
#define TELESCOPE_HPP

#include "Config.hpp"         // for Config
#include "factory.hpp"        // for FACTORY, CREATE_FACTORY, REGISTER_NAMED_TYPE_WITH_FACTORY
#include "kotekanLogging.hpp" // for ERROR, kotekanLogging
#include "restServer.hpp"     // for connectionInstance
#include "timeUtil.hpp"       // for EOP

#include "fmt.hpp" // for compile_string_to_view

#include <exception>    // for exception
#include <memory>       // for unique_ptr
#include <shared_mutex> // for shared_mutex
#include <stdint.h>     // for uint32_t, uint64_t, UINT32_MAX, uint8_t
#include <string>       // for string, basic_string
#include <time.h>       // for timespec

// Create the abstract factory for generating patterns
class Telescope;

CREATE_FACTORY(Telescope, const kotekan::Config&, const std::string&);
#define REGISTER_TELESCOPE(TelescopeType, name)                                                    \
    REGISTER_NAMED_TYPE_WITH_FACTORY(Telescope, TelescopeType, name)


using nyquist_zone_t = std::uint8_t;
using freq_id_t = std::uint32_t; // logical ID, not necessarily an index
#define FREQ_ID_NOT_SET UINT32_MAX


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
 * between sequence number and instrument time, and for returning sampling
 * parameters (raw sample rate, frequency spacing, map from freq_id to physical
 * frequency, etc).  It also contains the Earth Orientation Parameter (EOP) table,
 * which holds data to compute UT1 time and CIRS coordinate transformations from
 * instrument time.
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
     **/
    Telescope(const std::string& tel_path, const std::string& log_level, bool require_eop,
              const std::string& eop_updatable_config_path);

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
