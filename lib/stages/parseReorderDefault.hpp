#ifndef GIVEN_DATA_GEN_H
#define GIVEN_DATA_GEN_H

#include "Config.hpp"  // for Config
#include "Stage.hpp"   // for Stage
#include "buffer.hpp"  // for Buffer
#include "visUtil.hpp" // for input_ctype

#include <stdint.h> // for uint32_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class parseReorderDefault
 * @brief Parse the reordering configuration section
 *
 * @par Buffers
 * @buffer out_buf Buffer to with reorder table column
 *         @buffer_metadata chordMetadata
 *
 * @conf  name                  String. Name of the quantity being set.
 * @conf  input_reorder         Table. The input reorder table to parse.
 * @conf  invert_mapping        Bool. Default: true. Invert the input reorder
 *                              table, that is store the target indices instead
 *                              of the source indices consecutively.
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
    const std::vector<uint32_t> _input_reorder;
    const bool _invert_mapping;
    const int _num_polarizations;
    const int _num_dishes;
};

#endif
