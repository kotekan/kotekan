************
Processes
************

More detailed info about processes, how they should be structured, and what should go where.


Minimal Definition
-------------------
A minimalist process is defined in SampleProcess.cpp and reproduced here. This should be considered a minimal process.

.. code-block:: c++
   :linenos:

    #include "SampleProcess.hpp"
    #include "errors.h"

    SampleProcess::SampleProcess(Config &config, const string& unique_name,
                                 bufferContainer &buffer_container) :
        KotekanProcess(config, unique_name, buffer_container,
                       std::bind(&SampleProcess::main_thread, this)) {
    }

    SampleProcess::~SampleProcess() {
    }

    void SampleProcess::apply_config(uint64_t seq) {
        (void)seq;
    }

    void SampleProcess::main_thread() {
        INFO("Sample Process, reached main_thread!");
        while (!stop_thread) {
            INFO("In thread!");
        }
    }


