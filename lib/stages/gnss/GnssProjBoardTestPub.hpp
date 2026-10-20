/**
 * @file
 * @brief Test publisher of lane directions on the process-wide ProjBoard, so a CPU pipeline can
 *        run GnssN2Project with source lanes (#190).
 *  - GnssProjBoardTestPub : public kotekan::Stage
 */
#ifndef GNSS_PROJ_BOARD_TEST_PUB_HPP
#define GNSS_PROJ_BOARD_TEST_PUB_HPP
#include "Config.hpp"
#include "Stage.hpp"
#include "bufferContainer.hpp"

#include <string>
#include <vector>

/**
 * @class GnssProjBoardTestPub
 * @brief Every period_ms, publish n_dirs unit directions (src 2, coherence 1, power n_dirs - j)
 *        for each freq_id under board_group, as the assemblers do for tracked PRNs. Direction j
 *        is the unit vector exp(i (j + 1) 0.37 e^2) / sqrt(n_elem) over the tracker element
 *        axis e, so the directions are distinct and spread over the array. No buffers.
 *
 * @conf board_group  String. The ProjBoard group GnssN2Project reads.
 * @conf freq_ids     Int list. Absolute freq_ids to publish for.
 * @conf n_elem       Int. The tracker element axis length (tracker_stations' size).
 * @conf n_dirs       Int. Directions per channel (2).
 * @conf period_ms    Int. Publish period (20).
 */
class GnssProjBoardTestPub : public kotekan::Stage {
public:
    GnssProjBoardTestPub(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container);
    ~GnssProjBoardTestPub() override = default;
    void main_thread() override;

private:
    std::string board_group;
    std::vector<int> freq_ids;
    int n_elem, n_dirs, period_ms;
};
#endif
