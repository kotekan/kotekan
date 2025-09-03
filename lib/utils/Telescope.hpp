#ifndef TELESCOPE_HPP
#define TELESCOPE_HPP

#include "Config.hpp"         // for Config
#include "buffer.hpp"         // for Buffer
#include "factory.hpp"        // for CREATE_FACTORY, Factory, REGISTER_NAMED_TYPE_WITH_FACTORY
#include "kotekanLogging.hpp" // for kotekanLogging

#include <memory>   // for unique_ptr
#include <stdint.h> // for uint32_t, uint64_t, UINT32_MAX
#include <string>   // for string
#include <time.h>   // for timespec


// Create the abstract factory for generating patterns
class Telescope;

CREATE_FACTORY(Telescope, const kotekan::Config&, const std::string&);
#define REGISTER_TELESCOPE(TelescopeType, name)                                                    \
    REGISTER_NAMED_TYPE_WITH_FACTORY(Telescope, TelescopeType, name)


using freq_id_t = uint32_t;
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
 *
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


    virtual ~Telescope() = default;

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
     * Get the frequency ID from the FPGA stream ID.
     *
     * @param  buf  The buffer object.
     * @param  ID   Index of the frame in the buffer.
     *
     * @returns     The integer frequency ID.
     **/
    virtual freq_id_t to_freq_id(const Buffer* buf, int ID) const;

    /**
     * Get the frequency ID from the FPGA stream ID.
     *
     * @param  buf  The buffer object.
     * @param  ID   Index of the frame in the buffer.
     * @param  ind  The index for the multifrequency stream.
     *
     * @returns     The integer frequency ID.
     **/
    virtual freq_id_t to_freq_id(const Buffer* buf, int ID, uint32_t ind) const;

    /**
     * Get the physical frequency in MHz of the specified freq ID.
     *
     * @param  freq_id  The frequency ID.
     *
     * @returns         The central frequency in MHz.
     **/
    virtual double to_freq(freq_id_t freq_id) const = 0;


    /**
     * Get the physical frequency in MHz of the specified channel.
     *
     * The baseclass implementation just calls
     * `to_freq(to_freq_id(args))`, override with a custom implementation
     * to save a function call.
     *
     * @param  args  Any arguments accepted by `to_freq_id`.
     *
     * @returns      The central frequency in MHz.
     **/
    template<typename... Args>
    double to_freq(Args... args) const {
        return to_freq(to_freq_id(args...));
    }


    /**
     * @brief Get the number of frequencies per stream.
     *
     * @return  The number of frequencies on a stream.
     **/
    virtual uint32_t num_freq_per_stream() const = 0;


    /**
     * @brief Get the total number of frequencies channels.
     *
     * This is the upper bound for freq_id.
     *
     * @return  The total number of frequency channels.
     **/
    virtual uint32_t num_freq() const = 0;

    /**
     * @brief Get the frequency width of a given channel.
     *
     * @return  The width of the frequency channel in MHz.
     **/
    virtual double freq_width(freq_id_t freq_id) const = 0;

    /**
     * @brief Get which Nyquist zone we are in.
     *
     * @return  The Nyquist zone.
     **/
    virtual uint8_t nyquist_zone() const = 0;

    /**
     * Convert a sequence number into a UNIX epoch time.
     *
     * @param  seq  The sequence number.
     *
     * @return  The corresponding UNIX time.
     **/
    virtual timespec to_time(uint64_t seq) const = 0;


    /**
     * @brief Convert a UNIX epoch time into the nearest sequence number.
     *
     * @note When there is not an exact correspondence between the given time
     *       and FPGA sequence numbers, this routine will return the latest valid
     *       FPGA sequence number before the given timestamp.
     *
     * @param  time  The UNIX time.
     *
     * @return  The corresponding sequence number.
     **/
    virtual uint64_t to_seq(timespec time) const = 0;


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

private:
    static inline std::unique_ptr<Telescope> tel_instance = nullptr;

protected:
    /**
     * This constructor sets up the logging. Implement a specific constructor
     * in a derived class to parse the config, and call this one to make sure
     * the logging is done correctly.
     **/
    Telescope(const std::string& log_level);
};

#endif // TELESCOPE_HPP
