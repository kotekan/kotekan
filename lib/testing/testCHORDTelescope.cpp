#include "testCHORDTelescope.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO
#include "CHORDTelescope.hpp" // for CHORDTelescope
#include "timeUtil.hpp" // for CHORDTelescope
#include "errors.h" // for exit_kotekan

#include <atomic>     // for atomic_bool
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stdint.h>   // for uint32_t, uint8_t
#include <chrono>
#include <thread>

// Include the classes we will be using
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

// Register the stage with the stage factory.
REGISTER_KOTEKAN_STAGE(TestCHORDTelescope);

/*
 * TestCHORDTelescope constructor.  Note that you can instead use the macro
 *
 *    STAGE_CONSTRUCTOR(TestCHORDTelescope)
 *
 * which saves the boilerplate of the constructor signature.
 */
TestCHORDTelescope::TestCHORDTelescope(Config& config,
                                const std::string& unique_name,
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
                tel.get_tel_orientation_el(0, 0),
                tel.get_tel_orientation_el(0, 1),
                tel.get_tel_orientation_el(0, 2));
        INFO("                                   {0:.6f} {1:.6f} {2:.6f}",
                tel.get_tel_orientation_el(1, 0),
                tel.get_tel_orientation_el(1, 1),
                tel.get_tel_orientation_el(1, 2));
        INFO("                                   {0:.6f} {1:.6f} {2:.6f}",
                tel.get_tel_orientation_el(2, 0),
                tel.get_tel_orientation_el(2, 1),
                tel.get_tel_orientation_el(2, 2));
        INFO("            Dish Orientation: {0:.6f} {1:.6f} {2:.6f}",
                tel.get_dish_orientation_el(0, 0),
                tel.get_dish_orientation_el(0, 1),
                tel.get_dish_orientation_el(0, 2));
        INFO("                              {0:.6f} {1:.6f} {2:.6f}",
                tel.get_dish_orientation_el(1, 0),
                tel.get_dish_orientation_el(1, 1),
                tel.get_dish_orientation_el(1, 2));
        INFO("                              {0:.6f} {1:.6f} {2:.6f}",
                tel.get_dish_orientation_el(2, 0),
                tel.get_dish_orientation_el(2, 1),
                tel.get_dish_orientation_el(2, 2));
        int n_eop = tel.get_EOP_table_len();

        std::vector<timespec> eop_times;

        int i;
        INFO("            EOP entries: {:d}", n_eop);
        for(i=0; i<n_eop; i++) {
            struct EOP eop = tel.get_EOP_at_idx(i);
            eop_times.push_back(eop.t_inst);
            INFO("            {0: 2d} - t_inst: {1:d} s + {2:d} ns",
                    i, eop.t_inst.tv_sec, eop.t_inst.tv_nsec);
            INFO("               - t_ut1:  {0:d} s + {1:d} ns",
                    eop.t_ut1.tv_sec, eop.t_ut1.tv_nsec);
            INFO("               - dut1:   {:f} s", eop.delta_UT1_inst);
            INFO("               - era:    {:f} deg", eop.ERA_deg);
            INFO("               - xp:     {:f} arcsec", eop.xp_as);
            INFO("               - yp:     {:f} arcsec", eop.yp_as);
        }

        int nt = eop_times.size();

        INFO("            EOP Probes:");

        for(i=0; i <= nt; i++)
        {
            timespec ta, tb;
            if(i==0 && nt == 1)
                ta = {eop_times[0].tv_sec - 43200, 0};
            else if(i==0)
                ta = {eop_times[0].tv_sec
                        - 2*(eop_times[1].tv_sec-eop_times[0].tv_sec), 0};
            else
                ta = eop_times[i-1];

            if(i == nt && nt == 1)
                tb = {eop_times[0].tv_sec + 43200, 0};
            else if(i == nt)
                tb = {eop_times[nt-1].tv_sec
                        + 2*(eop_times[nt-1].tv_sec - eop_times[nt-2].tv_sec),
                      0};
            else
                tb = eop_times[i];
                

            int n_seg = 4;
            int64_t dns = 1'000'000'000L * (tb.tv_sec - ta.tv_sec)
                             + tb.tv_nsec - ta.tv_nsec;
            for(int j = 0; j<n_seg; j++)
            {
                int64_t ns = (j*dns)/n_seg;
                int64_t s = ns / 1'000'000'000;
                ns = ns - 1'000'000'000*s;

                if(ns + ta.tv_nsec >= 1'000'000'000) {
                    ns -= 1'000'000'000;
                    s += 1;
                }

                timespec t = {ta.tv_sec + s, ta.tv_nsec + ns};

                struct EOP eop = tel.get_EOP_at_time(t);
                INFO("               - t_inst: {1:d} s + {2:d} ns",
                        i, eop.t_inst.tv_sec, eop.t_inst.tv_nsec);
                INFO("               - t_ut1:  {0:d} s + {1:d} ns",
                        eop.t_ut1.tv_sec, eop.t_ut1.tv_nsec);
                INFO("               - dut1:   {:f} s", eop.delta_UT1_inst);
                INFO("               - era:    {:f} deg", eop.ERA_deg);
                INFO("               - xp:     {:f} arcsec", eop.xp_as);
                INFO("               - yp:     {:f} arcsec", eop.yp_as);

                timespec t_inst2 = get_time_from_UT1(eop.t_ut1, eop.delta_UT1_inst);
                int64_t n_rot;
                double era = get_ERA_from_UT1(eop.t_ut1, &n_rot);
                timespec t_ut12 = get_UT1_from_ERA(n_rot, eop.ERA_deg);

                INFO("               -t_inst2: {0:d} s + {1:d} ns",
                        t_inst2.tv_sec, t_inst2.tv_nsec);
                INFO("               -diff:    {0:d} s + {1:d} ns",
                        t_inst2.tv_sec - eop.t_inst.tv_sec,
                        t_inst2.tv_nsec - eop.t_inst.tv_nsec);
                INFO("               -t_ut12:  {0:d} s + {1:d} ns",
                        t_ut12.tv_sec, t_ut12.tv_nsec);
                INFO("               -diff:    {0:d} s + {1:d} ns",
                        t_ut12.tv_sec - eop.t_ut1.tv_sec,
                        t_ut12.tv_nsec - eop.t_ut1.tv_nsec);


                int64_t n_rot2;
                double era2 = get_ERA_from_UT1(t_ut12, &n_rot2);

                INFO("               -era:  {0:f} deg + {1:d}", era, n_rot);
                INFO("               -diff: {0:e} deg + {1:d}", era2-era,
                        n_rot2 - n_rot);



            }
        }

        int n_dish = tel.get_num_dishes();
        INFO("            Num Dishes:  {:d}", n_dish); 
        for(i=0; i<n_dish; i++) {
            std::array<double, 3> pos = tel.get_dish_position(i);
            INFO("            Dish Pos:    {0:d} - ({1:f}, {2:f}, {3:f})",
                 i, pos[0], pos[1], pos[2]);
        }

        //break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }

    exit_kotekan(CLEAN_EXIT);
}
