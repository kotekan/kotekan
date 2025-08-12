#include "testCHORDTelescope.hpp"

#include "CHORDTelescope.hpp" // for CHORDTelescope
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "errors.h"           // for exit_kotekan
#include "kotekanLogging.hpp" // for INFO
#include "timeUtil.hpp"       // for CHORDTelescope

#include <atomic> // for atomic_bool
#include <chrono>
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stdint.h>   // for uint32_t, uint8_t
#include <thread>

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

#define GIGA 1'000'000'000L

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(TestCHORDTelescope);

/*
 * TestCHORDTelescope constructor.  Note that you can instead use the macro
 *
 *    STAGE_CONSTRUCTOR(TestCHORDTelescope)
 *
 * which saves the boilerplate of the constructor signature.
 */
TestCHORDTelescope::TestCHORDTelescope(Config& config, const std::string& unique_name,
                                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&TestCHORDTelescope::main_thread, this)) {}

TestCHORDTelescope::~TestCHORDTelescope() {}

// Framework managed pthread
void TestCHORDTelescope::main_thread() {
    // Logging function
    INFO("Reached main_thread!");

    const CHORDTelescope& tel = Telescope::instance().cast<CHORDTelescope>();

    // Until the thread is stopped
    while (!stop_thread) {

        timespec t0 = tel.to_time(0);

        // Logging
        INFO("Tel type label: {}", tel.get_name());
        INFO("CHORD Tel - GPS enabled: {:d}", tel.gps_time_enabled());
        INFO("            time0:       {:d} s + {:d} ns", t0.tv_sec, t0.tv_nsec);

        double lat = tel.get_inst_lat_deg();
        double lon = tel.get_inst_long_deg();
        INFO("            lat:         {:f} deg", lat);
        INFO("            long:        {:f} deg", lon);
        INFO("            Telescope Orientation: {0:.6f} {1:.6f} {2:.6f}",
             tel.get_tel_orientation_el(0, 0), tel.get_tel_orientation_el(0, 1),
             tel.get_tel_orientation_el(0, 2));
        INFO("                                   {0:.6f} {1:.6f} {2:.6f}",
             tel.get_tel_orientation_el(1, 0), tel.get_tel_orientation_el(1, 1),
             tel.get_tel_orientation_el(1, 2));
        INFO("                                   {0:.6f} {1:.6f} {2:.6f}",
             tel.get_tel_orientation_el(2, 0), tel.get_tel_orientation_el(2, 1),
             tel.get_tel_orientation_el(2, 2));
        INFO("            Dish Orientation: {0:.6f} {1:.6f} {2:.6f}",
             tel.get_dish_orientation_el(0, 0), tel.get_dish_orientation_el(0, 1),
             tel.get_dish_orientation_el(0, 2));
        INFO("                              {0:.6f} {1:.6f} {2:.6f}",
             tel.get_dish_orientation_el(1, 0), tel.get_dish_orientation_el(1, 1),
             tel.get_dish_orientation_el(1, 2));
        INFO("                              {0:.6f} {1:.6f} {2:.6f}",
             tel.get_dish_orientation_el(2, 0), tel.get_dish_orientation_el(2, 1),
             tel.get_dish_orientation_el(2, 2));
        int n_eop = tel.get_EOP_table_len();

        std::vector<int64_t> eop_times;

        int i;
        INFO("            EOP entries: {:d}", n_eop);
        for (i = 0; i < n_eop; i++) {
            struct EOP eop = tel.get_EOP_at_idx(i);
            eop_times.push_back(eop.t_inst);
            INFO("            {0: 2d} - t_inst: {1:d} s + {2:d} ns", i, eop.t_inst / GIGA,
                 eop.t_inst % GIGA);
            INFO("               - t_ut1:  {0:d} s + {1:d} ns", eop.t_ut1 / GIGA, eop.t_ut1 % GIGA);
            INFO("               - dut1:   {:f} s", eop.delta_UT1_inst);
            INFO("               - era:    {:f} deg", eop.ERA_deg);
            INFO("               - xp:     {:f} arcsec", eop.xp_as);
            INFO("               - yp:     {:f} arcsec", eop.yp_as);
        }

        int nt = eop_times.size();

        INFO("            EOP Probes:");

        for (i = 0; i <= nt; i++) {
            int64_t ta, tb;
            if (i == 0 && nt == 1)
                ta = eop_times[0] - 43200 * GIGA;
            else if (i == 0)
                ta = eop_times[0] - 2 * (eop_times[1] - eop_times[0]);
            else
                ta = eop_times[i - 1];

            if (i == nt && nt == 1)
                tb = eop_times[0] + 43200 * GIGA;
            else if (i == nt)
                tb = eop_times[nt - 1] + 2 * (eop_times[nt - 1] - eop_times[nt - 2]);
            else
                tb = eop_times[i];


            int n_seg = 4;
            int64_t dns = tb - ta;
            for (int j = 0; j < n_seg; j++) {

                int64_t t = ta + (j * dns) / n_seg;
                timespec ts = {.tv_sec = (time_t)(t / GIGA), .tv_nsec = t % GIGA};

                struct EOP eop = tel.get_EOP_at_time(ts);
                INFO("               - t_inst: {1:d} s + {2:d} ns", i, eop.t_inst / GIGA,
                     eop.t_inst % GIGA);
                INFO("               - t_ut1:  {0:d} s + {1:d} ns", eop.t_ut1 / GIGA,
                     eop.t_ut1 % GIGA);
                INFO("               - dut1:   {:f} s", eop.delta_UT1_inst);
                INFO("               - era:    {:f} deg", eop.ERA_deg);
                INFO("               - xp:     {:f} arcsec", eop.xp_as);
                INFO("               - yp:     {:f} arcsec", eop.yp_as);

                timespec ts_inst2 = get_time_from_UT1(eop.t_ut1, eop.delta_UT1_inst);
                int64_t n_rot;
                double era = get_ERA_from_UT1(eop.t_ut1, &n_rot);
                int64_t t_ut12 = get_UT1_from_ERA(n_rot, eop.ERA_deg);

                INFO("               -t_inst2: {0:d} s + {1:d} ns", ts_inst2.tv_sec,
                     ts_inst2.tv_nsec);
                INFO("               -diff:    {0:d} s + {1:d} ns",
                     ts_inst2.tv_sec - eop.t_inst / GIGA, ts_inst2.tv_nsec - eop.t_inst % GIGA);
                INFO("               -t_ut12:  {0:d} s + {1:d} ns", t_ut12 / GIGA, t_ut12 % GIGA);
                INFO("               -diff:    {0:d} s + {1:d} ns", (t_ut12 - eop.t_ut1) / GIGA,
                     (t_ut12 - eop.t_ut1) % GIGA);


                int64_t n_rot2;
                double era2 = get_ERA_from_UT1(t_ut12, &n_rot2);

                INFO("               -era:  {0:f} deg + {1:d}", era, n_rot);
                INFO("               -diff: {0:e} deg + {1:d}", era2 - era, n_rot2 - n_rot);
            }
        }

        int n_dish = tel.get_num_dishes();
        INFO("            Num Dishes:  {:d}", n_dish);
        for (i = 0; i < n_dish; i++) {
            std::array<double, 3> pos = tel.get_dish_position(i);
            INFO("            Dish Pos:    {0:d} - ({1:f}, {2:f}, {3:f})", i, pos[0], pos[1],
                 pos[2]);
        }

        // break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }

    exit_kotekan(CLEAN_EXIT);
}
