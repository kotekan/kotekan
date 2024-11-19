#include "testCHORDTelescope.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO
#include "CHORDTelescope.hpp" // for CHORDTelescope
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
        
        double lat = tel.get_inst_lat();
        double lon = tel.get_inst_long();
        INFO("            lat:         {:f} deg", lat);
        INFO("            long:        {:f} deg", lon);
        INFO("            Orientation: {0:.6f} {1:.6f} {2:.6f}",
                tel.get_orientation_el(0, 0),
                tel.get_orientation_el(0, 1),
                tel.get_orientation_el(0, 2));
        INFO("                         {0:.6f} {1:.6f} {2:.6f}",
                tel.get_orientation_el(1, 0),
                tel.get_orientation_el(1, 1),
                tel.get_orientation_el(1, 2));
        INFO("                         {0:.6f} {1:.6f} {2:.6f}",
                tel.get_orientation_el(2, 0),
                tel.get_orientation_el(2, 1),
                tel.get_orientation_el(2, 2));
        INFO("            DUT1:        {:f} s", tel.get_dut1());
        INFO("            DTAI:        {:f} s", tel.get_dtai());
        int n_dish = tel.get_num_dishes();
        int i;
        INFO("            Num Dishes:  {:d}", n_dish); 
        for(i=0; i<n_dish; i++)
            INFO("            Dish Pos:    {0:d} - ({1:f}, {2:f}, {3:f})",
                 i, tel.get_dish_coord(i, 0),
                 tel.get_dish_coord(i, 1), tel.get_dish_coord(i, 2));

        //break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }

    exit_kotekan(CLEAN_EXIT);
}
