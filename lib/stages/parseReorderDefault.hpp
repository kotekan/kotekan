#ifndef GIVEN_DATA_GEN_H
#define GIVEN_DATA_GEN_H

#include <stdint.h>             // for uint32_t
#include <string>               // for string, basic_string
#include <vector>               // for vector

#include "Config.hpp"           // for Config
#include "Stage.hpp"            // for Stage
#include "Telescope.hpp"        // for ElementOrder
#include "buffer.hpp"           // for Buffer
#include "bufferContainer.hpp"  // for bufferContainer
#include "visUtil.hpp"          // for input_ctype

/**
 * @class parseReorderDefault
 * @brief Parse the reordering configuration section and create a mapping
 * yielding elements in output_order when indexed in input_order.
 *
 * @par Buffers
 * @buffer out_buf Buffer to with reorder table column
 *         @buffer_metadata chordMetadata
 *
 * @conf  name                  String. Name of the quantity being set.
 * @conf  input_order           ElementOrder. Default: CHIMECorrelator. Must be one of CHIMECorrelator, CHIMECylinder, CHIMEBeamformer.
 * @conf  output_order          ElementOrder. Default: CHIMECylinder. Must be one of CHIMECorrelator, CHIMECylinder, CHIMEBeamformer.
 * @conf  num_polarizations     Int. Number of polarizations. Used only to
 *                              compute number of elements.
 * @conf  num_dishes            Int. Number of dishes in telescope. Used only to
 *                              compute number of elements.
 *
 * @author Roland Haas
 */
class parseReorderDefault : public kotekan::Stage {
public:
    parseReorderDefault(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~parseReorderDefault() = default;
    void main_thread() override;

private:
    Buffer* const _out_buf;
    const std::string _name;
    const ElementOrder _input_order;
    const ElementOrder _output_order;
    const int _num_polarizations;
    const int _num_dishes;
    static const int _num_chime_cylinders = 4;
};

#endif
