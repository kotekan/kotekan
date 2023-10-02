/**
 * @file clBeamformPhaseData.hpp
 * @brief class to calculate phase data for beamforming.
 *  - clBeamformPhaseData: public clCommand
 */

#ifndef CL_BEAMFORM_PHASE_DATA_H
#define CL_BEAMFORM_PHASE_DATA_H

#include "clCommand.hpp"

#include <vector>

/**
 * @class clBeamformPhaseData
 * @brief This class is responsible for calculating phase shifts for every feed
 *        of a given telescope every duration set by a period.
 *
 * This code calculates phase information for the beamforming kernel. The phases are
 * written to the gpu memory buffer on every execute() call of this code and
 * expects a beamforming kernel enqueued after the memory to consume the phase data.
 *
 * @conf    beamforming_do_not_track    int (default 1). Set local beamform time to given time or
 * time first packet received.
 * @conf    inst_lat                    double (default 49.3203). Pathfinder latitude coordinate in
 * degrees.
 * @conf    inst_long                   double (default -119.6175). Pathfinder longitude coordinate
 * in degrees.
 * @conf    fixed_time                  int (default 1441828309). Given unix time to use for
 * beamform local time. If 0, current time used.
 * @conf    ra                          double. Right Ascension in degrees of source to beamform on.
 * @conf    dec                         double. Declination in degrees of source to beamform on.
 * @conf    feed_positions              float array. The x,y positions in meters of the feeds across
 * the telescope. pathfinder values: [-0.38928000000000001, 11.226240000000001, -0.37872, 10.92164,
 * -0.36815999999999999, 10.61702, -0.35759000000000002, 10.3124, -0.34703000000000001, 10.00778,
 * -0.33646999999999999, 9.7031500000000008, -0.32590999999999998, 9.3985500000000002,
 * -0.31534000000000001, 9.0939300000000003, -0.30478, 8.78932,
 * -0.29421999999999998, 8.4847000000000001, -0.28365000000000001, 8.1800800000000002,
 * -0.27309, 7.87547, -0.26252999999999999, 7.5708500000000001,
 * -0.25196000000000002, 7.2662300000000002, -0.2414, 6.9616199999999999,
 * -0.23083999999999999, 6.657, -0.22028, 6.3523800000000001,
 * -0.20971000000000001, 6.0477499999999997, -0.19914999999999999, 5.74315,
 * -0.18859000000000001, 5.4385300000000001, -0.17802000000000001, 5.1339100000000002,
 * -0.16746, 4.8292999999999999, -0.15690000000000001, 4.52468, -0.14634, 4.2200600000000001,
 * -0.13577, 3.9154499999999999, -0.12520999999999999, 3.61083, -0.11465, 3.3062100000000001,
 * -0.10408000000000001, 3.0015999999999998, -0.093520000000000006, 2.6969799999999999,
 * -0.082960000000000006, 2.39236, -0.072389999999999996, 2.0877500000000002,
 * -0.061830000000000003, 1.7831300000000001, -0.051270000000000003, 1.47851,
 * -0.040710000000000003, 1.1738900000000001, -0.03014, 0.86928000000000005, -0.01958,
 * 0.56466000000000005, -0.0090200000000000002, 0.26003999999999999, 0.0015499999999999999,
 * -0.044569999999999999, 0.012109999999999999, -0.34919, 0.022669999999999999, -0.65381,
 * 0.033230000000000003, -0.95842000000000005, 0.043799999999999999, -1.2630399999999999,
 * 0.054359999999999999, -1.5676600000000001, 0.064920000000000005, -1.8722700000000001,
 * 0.075490000000000002, -2.1768900000000002, 0.086050000000000001, -2.4815100000000001,
 * 0.096610000000000001, -2.78613, 0.10717, -3.0907399999999998, 0.11774, -3.3953600000000002,
 * 0.1283, -3.69998, 0.13886000000000001, -4.0045900000000003, 0.14943000000000001,
 * -4.3092100000000002, 0.15998999999999999, -4.6138300000000001, 0.17055000000000001,
 * -4.9184400000000004, 0.18112, -5.2230600000000003, 0.19167999999999999, -5.5276800000000001,
 * 0.20224, -5.8322900000000004, 0.21279999999999999, -6.1369100000000003, 0.22337000000000001,
 * -6.4415300000000002, 0.23393, -6.7461500000000001, 0.24449000000000001, -7.0507600000000004,
 * 0.25506000000000001, -7.3553800000000003, 0.26562000000000002, -7.6600000000000001,
 * 0.27617999999999998, -7.9646100000000004, -0.38928000000000001, 10.91107, -0.37872, 10.92164,
 * -0.36815999999999999, 10.61702, -0.35759000000000002, 10.3124, -0.34703000000000001, 10.00778,
 * -0.33646999999999999, 9.7031700000000001, -0.32590999999999998, 9.3985500000000002,
 * -0.31534000000000001, 9.0939300000000003, -0.30478, 8.78932,
 * -0.29421999999999998, 8.4847000000000001, -0.28365000000000001, 8.1800800000000002,
 * -0.27309, 7.87547, -0.26252999999999999, 7.5708500000000001,
 * -0.25196000000000002, 7.2662300000000002, -0.2414, 6.9616199999999999,
 * -0.23083999999999999, 6.657, -0.22028, 6.3523800000000001,
 * -0.20971000000000001, 6.0477699999999999, -0.19914999999999999, 5.74315,
 * -0.18859000000000001, 5.4385300000000001, -0.17802000000000001, 5.1339100000000002,
 * -0.16746, 4.8292999999999999, -0.15690000000000001, 4.52468, -0.14634, 4.2200600000000001,
 * -0.13577, 3.9154499999999999, -0.12520999999999999, 3.61083, -0.11465, 3.3062100000000001,
 * -0.10408000000000001, 3.0015999999999998, -0.093520000000000006, 2.6969599999999998,
 * -0.082960000000000006, 2.39236, -0.072389999999999996, 2.0877500000000002,
 * -0.061830000000000003, 1.7831300000000001, -0.051270000000000003, 1.47851,
 * -0.040710000000000003, 1.1738900000000001, -0.03014, 0.86928000000000005, -0.01958,
 * 0.56466000000000005, -0.0090200000000000002, 0.26003999999999999, 0.0015499999999999999,
 * -0.044569999999999999, 0.012109999999999999, -0.34919, 0.022669999999999999, -0.65381,
 * 0.033230000000000003, -0.95842000000000005, 0.043799999999999999, -1.2630399999999999,
 * 0.054359999999999999, -1.5676600000000001, 0.064920000000000005, -1.8722700000000001,
 * 0.075490000000000002, -2.1768900000000002, 0.086050000000000001, -2.4815100000000001,
 * 0.096610000000000001, -2.78613, 0.10717, -3.0907399999999998, 0.11774, -3.3953600000000002,
 * 0.1283, -3.69998, 0.13886000000000001, -4.0045900000000003, 0.14943000000000001,
 * -4.3092100000000002, 0.15998999999999999, -4.6138300000000001, 0.17055000000000001,
 * -4.9184400000000004, 0.18112, -5.2230600000000003, 0.19167999999999999, -5.5276800000000001,
 * 0.20224, -5.8323099999999997, 0.21279999999999999, -6.1369100000000003, 0.22337000000000001,
 * -6.4415300000000002, 0.23393, -6.7461500000000001, 0.24449000000000001, -7.0507600000000004,
 * 0.25506000000000001, -7.3553800000000003, 0.26562000000000002, -7.6600000000000001,
 * 0.27617999999999998,
 * -7.9646100000000004, 21.595610000000001, 12.043240000000001, 21.606169999999999, 11.738619999999999,
 * 21.61674, 11.433999999999999, 21.627300000000002, 11.129390000000001, 21.637889999999999, 10.823969999999999,
 * 21.64845, 10.519349999999999, 21.659020000000002, 10.214740000000001, 21.66958, 9.9101199999999992,
 * 21.68017, 9.6046999999999993, 21.690729999999999, 9.3000900000000009, 21.70129, 8.9954699999999992,
 * 21.711860000000001, 8.6908499999999993, 21.722100000000001, 8.3954299999999993, 21.732659999999999,
 * 8.0908099999999994, 21.743230000000001, 7.7862, 21.753789999999999, 7.4815800000000001, 21.76473, 7.1661700000000002,
 * 21.775289999999998, 6.8615300000000001, 21.78585, 6.5569300000000004, 21.796420000000001, 6.2523200000000001,
 * 21.807009999999998, 5.9469000000000003, 21.81757, 5.6422800000000004, 21.828130000000002, 5.3376700000000001,
 * 21.838699999999999, 5.0330500000000002, 21.84929, 4.7276300000000004, 21.859850000000002, 4.4230200000000002,
 * 21.87041, 4.1184000000000003, 21.880980000000001, 3.8137799999999999, 21.891220000000001, 3.5183599999999999,
 * 21.901779999999999, 3.21374, 21.91235, 2.9091300000000002, 21.922910000000002, 2.6045099999999999,
 * 21.933499999999999, 2.2990900000000001, 21.94406, 1.99448, 21.954630000000002, 1.6898599999999999,
 * 21.96519, 1.38524, 21.97578, 1.0798300000000001, 21.986339999999998,
 * 0.77520999999999995, 21.9969, 0.47059000000000001, 22.007470000000001,
 * 0.16597999999999999, 22.018059999999998, -0.13944000000000001, 22.02862,
 * -0.44406000000000001, 22.039180000000002, -0.74868000000000001, 22.04975,
 * -1.0532900000000001, 22.06034, -1.3587100000000001, 22.070900000000002, -1.66333, 22.08146,
 * -1.96794, 22.092030000000001, -2.2725599999999999, 22.102620000000002,
 * -2.5779800000000002, 22.11318, -2.88259, 22.123740000000002,
 * -3.1872099999999999, 22.134309999999999, -3.4918300000000002, 22.1449,
 * -3.7972399999999999, 22.155460000000001, -4.1018600000000003, 22.16602,
 * -4.4064800000000002, 22.176590000000001, -4.7110900000000004, 22.187180000000001,
 * -5.0165100000000002, 22.19774, -5.3211300000000001, 22.208300000000001,
 * -5.6257400000000004, 22.218859999999999, -5.9303600000000003, 22.22946,
 * -6.2357800000000001, 22.240020000000001, -6.5403900000000004, 22.250579999999999,
 * -6.8450300000000004, 22.261140000000001,
 * -7.1496300000000002, 21.595610000000001, 12.043240000000001, 21.606169999999999, 11.738619999999999,
 * 21.61674, 11.433999999999999, 21.627300000000002, 11.129390000000001, 21.637889999999999, 10.823969999999999,
 * 21.64845, 10.519349999999999, 21.659020000000002, 10.214740000000001, 21.66958, 9.9101199999999992,
 * 21.68017, 9.6046999999999993, 21.690729999999999, 9.3000900000000009, 21.70129, 8.9954699999999992,
 * 21.711860000000001, 8.6908499999999993, 21.722100000000001, 8.3954299999999993, 21.732659999999999,
 * 8.0908099999999994, 21.743230000000001, 7.7862, 21.753979999999999, 7.4707699999999999, 21.76473, 7.1661700000000002,
 * 21.775289999999998, 6.8615500000000003, 21.78585, 6.5569300000000004, 21.796420000000001, 6.2523200000000001,
 * 21.807009999999998, 5.9469000000000003, 21.81757, 5.6422800000000004, 21.828130000000002, 5.3376700000000001,
 * 21.838699999999999, 5.0330500000000002, 21.84929, 4.7276300000000004, 21.859850000000002, 4.4230200000000002,
 * 21.87041, 4.1184000000000003, 21.880980000000001, 3.8137799999999999, 21.891220000000001, 3.5183599999999999,
 * 21.901779999999999, 3.21373, 21.91235, 2.9091300000000002, 21.922920000000001, 2.6036899999999998,
 * 21.933499999999999, 2.2990900000000001, 21.94406, 1.99448, 21.954630000000002, 1.6898599999999999,
 * 21.96519, 1.38524, 21.97578, 1.0798300000000001, 21.986339999999998,
 * 0.77520999999999995, 21.9969, 0.47059000000000001, 22.007470000000001,
 * 0.16597999999999999, 22.018059999999998, -0.13944000000000001, 22.02862,
 * -0.44406000000000001, 22.039180000000002, -0.74868000000000001, 22.04975,
 * -1.0532900000000001, 22.06034, -1.3587100000000001, 22.070900000000002, -1.66333, 22.08146,
 * -1.96794, 22.092030000000001, -2.2725599999999999, 22.102620000000002,
 * -2.5779800000000002, 22.11318, -2.88259, 22.123740000000002,
 * -3.1872099999999999, 22.134309999999999, -3.4918300000000002, 22.1449,
 * -3.7972399999999999, 22.155460000000001, -4.1018600000000003, 22.16602,
 * -4.4064800000000002, 22.176590000000001, -4.7110900000000004, 22.187180000000001,
 * -5.0165100000000002, 22.19774, -5.3211300000000001, 22.208300000000001,
 * -5.6257400000000004, 22.218859999999999, -5.9303600000000003, 22.22946,
 * -6.2357800000000001, 22.240020000000001, -6.5403900000000004, 22.250579999999999,
 * -6.8450100000000003, 22.261140000000001, -7.14961]
 * @conf    num_elements                int (default 2048). Number of elements in telescope.
 *
 * @warning This code was written and tested on pathfinder. Migrating to another system will
 *          require careful inspection to ensure everything is working.
 *
 * @todo    The comments in the code need to be cleaned up. This code has been run many times and
 * seems to work based on the results, but further testing on another system is warranted to ensure
 * the values calculated here are correct.
 *
 * @author Ian Tretyakov
 *
 */

class clBeamformPhaseData : public clCommand {
public:
    /// Constructor, no logic added.
    clBeamformPhaseData(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, clDeviceInterface& device,
			int instance_num);
    /// Allocate size for phases arrays and initialize start_beamform_time and last_bankID.
    virtual void build() override;
    /// Enqueues a new array of phases on the gpu. Phases are updated every "phase period" (1
    /// second) by referencing a "phase bank" array that stores two arrays of phases and enqueues
    /// either based on the fpga sequence number of the current buffer and buffer_ID. (Sequence
    /// number for CHIME is 2.56us).
    virtual cl_event execute(cl_event pre_event) override;

protected:
    /** The get_delays method is called in execute to determine the current phase delays to store in
     *the "phase bank" bank array. The phases are calculated per feed to a source defined by ra and
     *dec relative to the instrument's latitude and longitude coordinates. The phases are
     *recalculated every phase period to account for the offset of the instrument relative to the
     *source due to the transit of the sky as determined by the local sidereal time accounting for
     *precession and nutation of the earth's rotation.
     *
     * @param phases        Pointer to a single index of phase delays stored in the phase bank
     *array. The array of delays are written to this pointer to be returned to the calling function.
     *
     * @param beamform_time The currently set beamform time determined in the execute method.
     **/
    void get_delays(float* phases, time_t beamform_time);

    // phase data
    /// The phase bank array storing two arrays of phases that are alternatively enqueued based on
    /// the fraction remaining from the incrementing fpga sequence divided by the phase period.
    float* phases[2];
    /// Sets local beamform time to given time or time first packet received.
    int beamforming_do_not_track;
    /// Pathfinder latitude coordinate in degrees.
    double inst_lat;
    /// Pathfinder longitude coordinate in degrees.
    double inst_long;
    /// Given unix time to use for beamform local time. If 0, current time used.
    int fixed_time;
    /// Right Ascension in degrees of source to beamform on.
    double ra;
    /// Declination in degrees of source to beamform on.
    double dec;
    /// The x,y positions in meters of the feeds across the telescope.
    std::vector<float> feed_positions;
    /// Set to current time in build() and pass that value to local beamform time if fixed_time is
    /// 0.
    time_t start_beamform_time;
    /// Initialized to -1 and vary the value between 1 or 2 to pass either index 1 or 2 of the phase
    /// bank bank array when the duration of the phase period has elapsed.
    int64_t last_bankID;
    /// Number of elements on the telescope (e.g. 2048 - CHIME, 256 - Pathfinder).
    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    Buffer* network_buf;
};

#endif // CL_BEAMFORM_PHASE_DATA_H
