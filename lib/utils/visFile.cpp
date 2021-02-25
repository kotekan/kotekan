
#include "visFile.hpp"

#include "fmt.hpp" // for format, fmt

#include <ctime>      // for gmtime, time_t
#include <fstream>    // for basic_ostream::operator<<, ofstream, endl, basic_ostream, basic_os...
#include <iterator>   // for reverse_iterator
#include <libgen.h>   // for dirname, basename
#include <sys/stat.h> // for mkdir
#include <unistd.h>   // for getpid


visFileBundle::~visFileBundle() {
    // Deactivate each open sample and remove them from the map
    auto it = vis_file_map.begin();

    while (it != vis_file_map.end()) {
        auto file = it->second.first;
        auto ind = it->second.second;
        file->deactivate_time(ind);
        it = vis_file_map.erase(it);
    }
}


time_ctype visFileBundle::last_update() const {
    return vis_file_map.rbegin()->first;
}


bool visFileBundle::resolve_sample(time_ctype new_time) {

    if (vis_file_map.size() == 0) {
        // If no files are currently in the map we should create a new one.
        add_file(new_time);
    } else {
        // If there are files in the list we need to figure out whether to
        // insert a new entry or not
        time_ctype max_time = vis_file_map.rbegin()->first;
        time_ctype min_time = vis_file_map.begin()->first;

        if (new_time < min_time) {
            // This data is older that anything else in the map so we should just drop it
            INFO("Dropping integration as buffer (FPGA count: {:d}) arrived too late (minimum in "
                 "pool {:d})",
                 new_time.fpga_count, min_time.fpga_count);
            return false;
        }

        if (new_time > max_time) {
            // We've got a later time and so we need to add a new time sample,
            // if the current file does not need to rollover register the new
            // sample as being in the last file, otherwise create a new file
            std::shared_ptr<visFile> file;
            uint32_t ind;
            std::tie(file, ind) = vis_file_map.rbegin()->second; // Unpack the last entry

            if ((rollover == 0 || file->num_time() < rollover) && !change_file) {
                // Extend the time axis and add into the sample map
                ind = file->extend_time(new_time);
                vis_file_map[new_time] = std::make_pair(file, ind);
            } else {
                add_file(new_time);
                change_file = false;
            }

            // As we've added a new sample we need to delete the earliest sample
            if (vis_file_map.size() > window_size) {
                std::tie(file, ind) = vis_file_map.begin()->second; // Unpack the first entry
                file->deactivate_time(ind);                         // Cleanup the sample
                vis_file_map.erase(vis_file_map.begin());
            }
        }
    }

    if (vis_file_map.find(new_time) == vis_file_map.end()) {
        // This is slightly subtle, but if a sample was not found at this point
        // then it must lie within the range, but not have been saved into the
        // files already. This means that adding it would make the files time
        // axis be out of order, so we just skip it for now.
        INFO("Skipping integration (FPGA count {:d}) as it would be written out of order.",
             new_time.fpga_count);
        return false;
    }

    return true;
}


void visFileBundle::add_file(time_ctype first_time) {

    time_t t = (time_t)first_time.ctime;

    // Start the acq and create the directory if required
    if (acq_name.empty()) {
        // Format the time (annoyingly you still have to use streams for this)
        acq_name = fmt::format("{:%Y%m%dT%H%M%SZ}_{:s}_{:s}", *std::gmtime(&t), instrument_name, acq_type);
        // Set the acq fields on the instance
        acq_start_time = first_time.ctime;

        // Create acquisition directory. Don't bother checking if it already exists, just let it
        // transparently fail
        mkdir((root_path + "/" + acq_name).c_str(), 0775);
    }

    // Construct the name of the new file
    uint32_t time_since_start = (uint32_t)(first_time.ctime - acq_start_time);
    std::string file_name;
    if(acq_type == "hfb") {
      file_name = fmt::format(fmt("{:08d}"), time_since_start);
    }
    else {
      file_name = fmt::format(fmt("{:08d}_{:04d}"), time_since_start, freq_chunk);
    }

    // Create the file, create room for the first sample and add into the file map
    auto file = mk_file(file_name, acq_name, root_path);
    auto ind = file->extend_time(first_time);
    vis_file_map[first_time] = std::make_pair(file, ind);
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
