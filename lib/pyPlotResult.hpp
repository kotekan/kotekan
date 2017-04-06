#ifndef PY_PLOT_RESULT_H
#define PY_PLOT_RESULT_H

#include "buffers.h"
#include "KotekanProcess.hpp"
#include "restServer.hpp"
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>

class pyPlotResult : public KotekanProcess {
public:
    pyPlotResult(Config &config,
                 struct Buffer &buf,
                 int gpu_id,
                 const std::string &base_dir,
                 const std::string &file_name,
                 const std::string &file_ext);
    virtual ~pyPlotResult();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread();
    void request_plot_callback(connectionInstance& conn, json& json_request);
private:
    struct Buffer &buf;
    std::string base_dir;
    std::string file_name;
    std::string file_ext;

    int gpu_id=-1;
    bool dump_plot=false;
};

#endif