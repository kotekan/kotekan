#include "GnssProjBoardTestPub.hpp"

#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE
#include "gnssProjSubspace.hpp" // for ProjBoard, ProjEntry
#include "kotekanLogging.hpp"   // for INFO

#include <chrono>
#include <cmath>
#include <complex>
#include <thread>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssProjBoardTestPub);

GnssProjBoardTestPub::GnssProjBoardTestPub(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssProjBoardTestPub::main_thread, this)),
    board_group(config.get<std::string>(unique_name, "board_group")),
    freq_ids(config.get<std::vector<int>>(unique_name, "freq_ids")),
    n_elem(config.get<int>(unique_name, "n_elem")),
    n_dirs(config.get_default<int>(unique_name, "n_dirs", 2)),
    period_ms(config.get_default<int>(unique_name, "period_ms", 20)) {
    if (n_elem < 1 || n_dirs < 1 || freq_ids.empty())
        FATAL_ERROR("GnssProjBoardTestPub: n_elem {:d}, n_dirs {:d}, {:d} freq_ids", n_elem, n_dirs,
                    (int)freq_ids.size());
    INFO("GnssProjBoardTestPub: {:d} directions x {:d} channels on {:s} every {:d} ms", n_dirs,
         (int)freq_ids.size(), board_group, period_ms);
}

void GnssProjBoardTestPub::main_thread() {
    gnss::ProjBoard& board = gnss::ProjBoard::instance();
    const float inv = 1.0f / std::sqrt((float)n_elem);
    while (!stop_thread) {
        const double now_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        for (int j = 0; j < n_dirs; ++j) {
            gnss::ProjEntry e;
            e.owner = unique_name + "/lane/" + std::to_string(j);
            e.src = 2;
            e.t_pub = now_s;
            e.power = (double)(n_dirs - j);
            e.coh = 1.0;
            e.n = n_elem;
            e.k = 1;
            // A held track at the zenith, so get_lanes serves the entry.
            e.los[2] = 1.0;
            e.t_utc =
                std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                    .count();
            e.q.reserve((size_t)n_elem);
            for (int el = 0; el < n_elem; ++el) {
                const float ph = (float)(j + 1) * 0.37f * (float)(el * el);
                e.q.emplace_back(inv * std::cos(ph), inv * std::sin(ph));
            }
            for (const int fid : freq_ids)
                board.publish(board_group, fid, e);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
    }
}
