#include "testCHORDTelescope.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "kotekanLogging.hpp" // for INFO
#include "CHORDTelescope.hpp" // for CHORDTelescope
#include "errors.h" // for exit_kotekan

#include <atomic>     // for atomic_bool
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stdint.h>   // for uint32_t, uint8_t

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
          std::bind(&TestCHORDTelescope::main_thread, this)) {

    // Check telescope is correct type.
    // This will print an informative error message, unlike the exception
    // thrown later.
    auto& tel = Telescope::instance();
    if (tel.get_name() != "CHORDTelescope")
    {
        ERROR("Telescope configured as: {}, not CHORDTelescope.",
                tel.get_name());
    }
}

TestCHORDTelescope::~TestCHORDTelescope() {}

// Framework managed pthread
void TestCHORDTelescope::main_thread() {
    // Logging function
    INFO("Reached main_thread!");

    const CHORDTelescope& tel = dynamic_cast<const CHORDTelescope&>(
                                    Telescope::instance());

    // Until the thread is stopped
    while (!stop_thread) {

        timespec t0 = tel.to_time(0);


        // Logging
        INFO("CHORD Tel - GPS enabled: {:d}", tel.gps_time_enabled());
        INFO("            time0:       {:d} s + {:d} ns", t0.tv_sec, t0.tv_nsec);
        
        double lat = tel.get_inst_lat();
        double lon = tel.get_inst_long();
        INFO("            lat:         {:f} deg", lat);
        INFO("            long:        {:f} deg", lon);
        INFO("Tel type name: {}", typeid(tel).name());
        INFO("Tel type label: {}", tel.get_name());

        break;
    }

    exit_kotekan(CLEAN_EXIT);
}
