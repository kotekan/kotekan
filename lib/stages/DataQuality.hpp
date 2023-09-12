/**
 * @file
 * @brief Compute a data quality metric for the visibility data.
 *  - DataQuality : public kotekan::Stage
 */

#ifndef DATA_QUALITY_STAGE
#define DATA_QUALITY_STAGE

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "datasetManager.hpp"    // for fingerprint_t, dset_id_t
#include "prometheusMetrics.hpp" // for Gauge, MetricFamily

#include <map>    // for map
#include <string> // for string
#include <vector> // for vector

/**
 * @class DataQuality
 * @brief Compute data quality metric for the visbility data, which is effectively
 *        the total amount of "data weight" per time and frequency."
 *
 * @par Buffers
 * @buffer in_buf Visibility data.
 *     @buffer_format VisBuffer structured
 *     @buffer_metadata VisMetadata
 *
 * @par Metrics
 * @metric kotekan_dataquality_dataquality
 *      The data quality of the visibility data.
 *
 * @author James Willis
 *
 */
class DataQuality : public kotekan::Stage {
public:
    /// Constructor.
    DataQuality(kotekan::Config& config_, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    /// Destructor
    virtual ~DataQuality();
    /// Primary loop to wait for buffers, dig through data,
    /// stuff packets lather, rinse and repeat.
    void main_thread() override;

private:
    /// Calculate a set of alpha coefficients for a given dataset
    void calc_alpha_coeffs(fingerprint_t fprint, dset_id_t ds_id);

    Buffer* in_buf;

    // Map the incoming fingerprint to a set of alpha coefficientse
    std::map<fingerprint_t, std::vector<double>> fprint_map;

    // Map the incoming datasets to fingerprints
    std::map<dset_id_t, fingerprint_t> dset_id_map;

    /// Prometheus metrics to export
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& data_quality_metric;
};

#endif
