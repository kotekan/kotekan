#ifndef TEST_DATA_CHECK_EXPECTED_HPP
#define TEST_DATA_CHECK_EXPECTED_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <stddef.h> // for size_t
#include <string>   // for string
#include <vector>   // for vector

/**
 * @class testDataCheckExpected
 * @brief Check float frames against expected constant values, known analytically.
 *
 * The expected values are cycled over each frame's floats, so a short list checks a
 * repeating pattern (e.g. one entry per component of the innermost axis). With a
 * second buffer, the elementwise difference (first - second) is checked instead --
 * useful when two pipelines differ by an analytically known amount while sharing an
 * unknown term (e.g. a bias) that cancels in the difference.
 *
 * Follows the testDataCheck conventions: after `num_frames_to_test` clean frames the
 * test passes (exiting kotekan once every checker has passed), and any mismatching
 * frame fails the test immediately.
 *
 * @par Buffers
 * @buffer first_buf   Float frames to check.
 *      @buffer_format float32
 * @buffer second_buf  Optional. When set, check (first_buf - second_buf) elementwise;
 *                     frame sizes must match.
 *      @buffer_format float32
 *
 * @conf  expected             Array<Float>. Values the frame (or difference) must
 *                             hold, cycled over each frame's floats.
 * @conf  skip_components      Array<Int>, default empty. Indices into the expected
 *                             cycle to skip (e.g. a component with no analytic value).
 * @conf  epsilon              Double, default 1e-6. Maximum absolute deviation.
 * @conf  num_frames_to_test   Int. Frames to check before declaring a pass.
 * @conf  max_num_errors_logged Int, default 100. Per-frame cap on logged mismatches.
 * @conf  trigger_exit_on_pass Bool, default true. Exit kotekan when every checker
 *                             has passed.
 *
 * @author James Mertens
 */
class testDataCheckExpected : public kotekan::Stage {
public:
    testDataCheckExpected(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    ~testDataCheckExpected() = default;
    void main_thread() override;

private:
    Buffer* first_buf;
    Buffer* second_buf; // null when only first_buf is checked
    std::vector<float> expected;
    std::vector<bool> check_component; // per expected-cycle index
    const double epsilon;
    const int num_frames_to_test;
    const int max_num_errors_logged;
    const bool trigger_exit_on_pass;
};

#endif // TEST_DATA_CHECK_EXPECTED_HPP
