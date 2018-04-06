
#include "visFile.hpp"
#include "errors.h"
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>


// Initialise the type map
std::map<std::string, std::function<visFile*()>> visFile::_type_list;


std::shared_ptr<visFile> visFile::create(
    const std::string& type,
    const std::string& name,
    const std::map<std::string, std::string>& metadata,
    const std::vector<freq_ctype>& freqs,
    const std::vector<input_ctype>& inputs,
    const std::vector<prod_ctype>& prods,
    size_t num_ev, size_t max_time
) {

    // Lookup the registered file and create an instance
    INFO("Creating file %s of type %s", name.c_str(), type.c_str());
    auto file = std::shared_ptr<visFile>(_type_list[type]());
    file->create_file(name, metadata, freqs, inputs, prods, num_ev, max_time);

    return file;
}


bool visFileBundle::resolve_sample(time_ctype new_time) {

    uint64_t count = new_time.fpga_count;

    if(vis_file_map.size() == 0) {
        // If no files are currently in the map we should create a new one.
        add_file(new_time);
    } else {
        // If there are files in the list we need to figure out whether to
        // insert a new entry or not
        uint64_t max_fpga = vis_file_map.rbegin()->first;
        uint64_t min_fpga = vis_file_map.begin()->first;

        if(count < min_fpga) {
            // This data is older that anything else in the map so we should just drop it
            INFO("Dropping integration as buffer (FPGA count: %" PRIu64
                 ") arrived too late (minimum in pool %" PRIu64 ")",
                 new_time.fpga_count, min_fpga);
            return false;
        }

        if(count > max_fpga) {
            // We've got a later time and so we need to add a new time sample,
            // if the current file does not need to rollover register the new
            // sample as being in the last file, otherwise create a new file
            std::shared_ptr<visFile> file;
            uint32_t ind;
            std::tie(file, ind) = vis_file_map.rbegin()->second;  // Unpack the last entry

            if(file->num_time() < rollover) {
                // Extend the time axis and add into the sample map
                ind = file->extend_time(new_time);
                vis_file_map[count] = std::make_tuple(file, ind);
            } else {
                add_file(new_time);
            }

            // As we've added a new sample we need to delete the earliest sample
            if(vis_file_map.size() > window_size) {
                vis_file_map.erase(vis_file_map.begin());
            }
        }
    }

    if(vis_file_map.find(count) == vis_file_map.end()) {
        // This is slightly subtle, but if a sample was not found at this point
        // then it must lie within the range, but not have been saved into the
        // files already. This means that adding it would make the files time
        // axis be out of order, so we just skip it for now.
        INFO("Skipping integration (FPGA count %" PRIu64
             ") as it would be written out of order.", count);
        return false;
    }

    return true;
}


void visFileBundle::add_file(time_ctype first_time) {

    time_t t = (time_t)first_time.ctime;

    // Start the acq and create the directory if required
    if(acq_name.empty()) {
        // Format the time (annoyingly you still have to use streams for this)
        std::ostringstream s;
        s << std::put_time(std::gmtime(&t), "%Y%m%dT%H%M%SZ");
        // Set the acq name
        acq_name = s.str() + "_" + instrument_name + "_corr";

        // Set the acq fields on the instance
        acq_start_time = first_time.ctime;

        // Create acquisition directory. Don't bother checking if it already exists, just let it transparently fail
        mkdir((root_path + "/" + acq_name).c_str(), 0755);
    }

    // Construct the name of the new file
    char fname_temp[100];
    snprintf(
        fname_temp, sizeof(fname_temp), "%08d_%04d.h5",
        (unsigned int)(first_time.ctime - acq_start_time), freq_chunk
    );
    std::string file_name = fname_temp;

    // Create the file, create room for the first sample and add into the file map
    auto file = mk_file(file_name, acq_name, root_path);
    auto ind = file->extend_time(first_time);
    vis_file_map[first_time.fpga_count] = std::make_tuple(file, ind);
}
