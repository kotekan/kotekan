
#include "visFile.hpp"
#include "errors.h"
#include "version.h"
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <libgen.h>
#include <fstream>
#include <sys/stat.h>
#include "fmt.hpp"


std::map<std::string, std::function<visFile*()>>& visFile::_registered_types()
{
    static std::map<std::string, std::function<visFile*()>> _register;

    return _register;
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

            if((rollover == 0 || file->num_time() < rollover) && !change_file) {
                // Extend the time axis and add into the sample map
                ind = file->extend_time(new_time);
                vis_file_map[count] = std::make_tuple(file, ind);
            } else {
                add_file(new_time);
                change_file = false;
            }

            // As we've added a new sample we need to delete the earliest sample
            if(vis_file_map.size() > window_size) {
                std::tie(file, ind) = vis_file_map.begin()->second;  // Unpack the first entry
                file->deactivate_time(ind); // Cleanup the sample
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
        acq_name = fmt::format("{:%Y%m%dT%H%M%SZ}_{}_corr",
                               *std::gmtime(&t), instrument_name);
        // Set the acq fields on the instance
        acq_start_time = first_time.ctime;

        // Create acquisition directory. Don't bother checking if it already exists, just let it transparently fail
        mkdir((root_path + "/" + acq_name).c_str(), 0755);
    }

    // Construct the name of the new file
    uint32_t time_since_start = (uint32_t)(first_time.ctime - acq_start_time);
    std::string file_name = fmt::format("{:08d}_{:04d}",
                                        time_since_start, freq_chunk);

    // Create the file, create room for the first sample and add into the file map
    auto file = mk_file(file_name, acq_name, root_path);
    auto ind = file->extend_time(first_time);
    vis_file_map[first_time.fpga_count] = std::make_tuple(file, ind);
}

void visCalFileBundle::set_file_name(std::string fname, std::string aname) {
    file_name = fname;
    acq_name = aname;
}

void visCalFileBundle::add_file(time_ctype first_time) {
    // Create directory
    mkdir((root_path + "/" + acq_name).c_str(), 0755);
    // Create the file, create room for the first sample and add into the file map
    auto file = mk_file(file_name, acq_name, root_path);
    auto ind = file->extend_time(first_time);
    vis_file_map[first_time.fpga_count] = std::make_tuple(file, ind);
}

void visCalFileBundle::swap_file(std::string new_fname, std::string new_aname) {
    // Change the file and and request writing to a new file
    set_file_name(new_fname, new_aname);
    change_file = true;
}

void visCalFileBundle::clear_file_map() {
    // RFlush and remove all entries in the map
    std::shared_ptr<visFile> file;
    uint32_t ind;
    for (size_t i = 0; i < vis_file_map.size(); i++) {
        std::tie(file, ind) = vis_file_map[i];
        file->deactivate_time(ind); // Cleanup the sample
    }
    vis_file_map.clear();
}

std::string create_lockfile(std::string filename) {

    // Create the lock file first such that there is no time the file is
    // unlocked
    std::string dir = filename;
    std::string base = filename;
    dir = dirname(&dir[0]);
    base = basename(&base[0]);

    std::string lock_filename = dir + "/." + base + ".lock";
    std::ofstream lock_file(lock_filename);
    lock_file << getpid() << std::endl;
    lock_file.close();

    return lock_filename;
}
