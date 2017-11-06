#include "hdf5Writer.hpp"
#include "util.h"
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include "gpuPostProcess.hpp"
#include "errors.h"
#include <time.h>
#include <algorithm>
#include <stdexcept>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>


const std::string FILE_NAME("testoutput.h5");
const std::string DATASET_NAME("vis");
const size_t BLOCK_SIZE = 64;

using namespace HighFive;


inline uint32_t cmap(uint32_t i, uint32_t j, uint32_t n) {
    return (n * (n + 1) / 2) - ((n - i) * (n - i + 1) / 2) + (j - i);
}

inline uint32_t prod_index(uint32_t i, uint32_t j, uint32_t block, uint32_t N) {
    uint32_t b_ix = cmap(i / block, j / block, N / block);

    return block * block * b_ix + i * block + j;

}


std::vector<complex_int> copy_vis_triangle(
    const complex_int * buf, const std::vector<uint32_t>& inputmap,
    size_t block, size_t N
) {

    size_t M = inputmap.size();
    std::vector<complex_int> output(M * (M + 1) / 2);
    size_t pi = 0;
    uint32_t bi;

    if(*std::max_element(inputmap.begin(), inputmap.end()) >= N) {
        throw std::invalid_argument("Input map asks for elements out of range.");
    }

    for(auto i = inputmap.begin(); i != inputmap.end(); i++) {
        for(auto j = i; j != inputmap.end(); j++) {
            bi = prod_index(*i, *j, block, N);
            output[pi] = buf[bi];
            pi++;
        }
    }

    return output;
}


hdf5Writer::hdf5Writer(Config& config,
                        const string& unique_name,
                        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&hdf5Writer::main_thread, this)) {

    buffers.push_back(get_buffer("buf"));

    for(auto buf : buffers) {
        register_consumer(buf, unique_name.c_str());
    }

    // Initialise the reordering mapping
    input_remap = std::vector<uint32_t>(2048);
    std::iota(input_remap.begin(), input_remap.end(), 0);

    // Temporarily fill out the input and freq vectors
    for(int i=0; i < 1024; i++) {
        freqs.push_back({800.0 - i * (400.0 / 1024), (400.0 / 1024)});
    }

    // Temporarily fill out the input and freq vectors
    for(int i=0; i < 2048; i++) {
        input_ctype t;
        t.chan_id = 0;
        std::string ts = "meh";
        ts.copy(t.correlator_input, 32);
        inputs.push_back(t);
    }

    num_elements = config.get_int("/", "num_elements");
}


void hdf5Writer::apply_config(uint64_t fpga_seq) {
}

void hdf5Writer::main_thread() {

    int frame_id = 0;
    uint8_t * frame = NULL;


    while (!stop_thread) {

        INFO("I am here");

        if (current_file == nullptr) {

            current_file = std::unique_ptr<visFile>(
                new visFile(FILE_NAME, freqs, inputs)
            );
            //current_file->createIndex(freqs, inputs);


        }

        // TODO: call a routine that returns a vector of all buffers that are
        // ready to read

        for(auto& buf : buffers) {

            frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);

            uint64_t fpga_seq = get_fpga_seq_num(buf, frame_id);
            stream_id_t stream_id = get_stream_id_t(buf, frame_id);
            timeval time_v = get_first_packet_recv_time(buf, frame_id);
            uint64_t lost_samples = get_lost_timesamples(buf, frame_id);

            char time_buf[64];
            time_t temp_time = time_v.tv_sec;
            struct tm* l_time = localtime(&temp_time);
            strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", l_time);

            INFO("Metadata for %s[%d]: FPGA Seq: %" PRIu64
                    ", stream ID = {create ID: %d, slot ID: %d, link ID: %d, freq ID: %d}, lost samples: %" PRIu64
                    ", time stamp: %ld.%06ld (%s.%06ld)",
                    buf->buffer_name, frame_id, fpga_seq,
                    stream_id.crate_id, stream_id.slot_id,
                    stream_id.link_id, stream_id.unused, lost_samples,
                    time_v.tv_sec, time_v.tv_usec, time_buf, time_v.tv_usec);

            // Construct the new time sample
            double dtime = (double)time_v.tv_sec + 1e-6 * time_v.tv_usec;
            time_ctype t = {fpga_seq, dtime};
            const std::vector<complex_int> vis; // = copy_vis_triangle(buf, input_remap, BLOCK_SIZE, num_elements);
            current_file->addSample(t, vis);


            mark_frame_empty(buf, unique_name.c_str(), frame_id);
            frame_id = (frame_id + 1) % buf->num_frames;




            //
        }
        // struct tcp_frame_header * header = (struct tcp_frame_header *)buf;
        //
        // int offset = sizeof(struct tcp_frame_header);
        // complex_int_t * visibilities = ( complex_int_t * )&buf[offset];
        //
        // offset += num_values * sizeof(complex_int_t);
        // struct per_frequency_data * frequency_data = ( struct per_frequency_data * )&buf[offset];
        //
        // offset += _num_total_freq * sizeof(struct per_frequency_data);
        // struct per_element_data * element_data = (struct per_element_data *)&buf[offset];
        //
        // offset += _num_total_freq * _num_elem *
        //           sizeof(struct per_element_data);
        // uint8_t * vis_weight = (uint8_t *)&buf[offset];
    }

    // close_file();

}


visFile::visFile(const std::string& name,
                 const std::vector<freq_ctype>& freqs,
                 const std::vector<input_ctype>& inputs) {

    // TODO: create lock file

    size_t ninput = inputs.size();

    file = std::unique_ptr<File>(
        new File(name, File::ReadWrite | File::Create | File::Truncate)
    );

    createIndex(freqs, inputs);
    createDatasets(freqs.size(), ninput, ninput * (ninput + 1) / 2);
}

// visFile::~visFile() {
//     file.flush();
//     delete time_imap;
//     delete vis;
//     delete file;
// }

void visFile::createIndex(const std::vector<freq_ctype>& freqs,
                          const std::vector<input_ctype>& inputs) {

    Group indexmap = file->createGroup("index_map");

    auto td = indexmap.createDataSet(
      "time", DataSpace({0}, {DataSpace::UNLIMITED}),
      create_datatype<time_ctype>(), std::vector<size_t>({1})
    );
    time_imap = std::unique_ptr<DataSet>(new DataSet(td));

    // Create and fill frequency dataset
    DataSet freq_imap = indexmap.createDataSet<freq_ctype>("freq", DataSpace(freqs.size()));
    freq_imap.write(freqs);


    DataSet input_imap = indexmap.createDataSet<input_ctype>("input", DataSpace(inputs.size()));


    std::vector<prod_ctype> prod_vector;
    for(uint16_t i=0; i < inputs.size(); i++) {
        for(uint16_t j = i; j < inputs.size(); j++) {
            prod_vector.push_back({i, j});
        }
    }
    DataSet prod_imap = indexmap.createDataSet<prod_ctype>(
        "prod", DataSpace(prod_vector.size())
    );
    prod_imap.write(prod_vector);

    file->flush();

}

void visFile::createDatasets(size_t nfreq, size_t ninput, size_t nprod) {

    // Create the visibility dataset
    DataSpace vis_space = DataSpace({0, nfreq, nprod},
                                    {DataSpace::UNLIMITED, nfreq, nprod});


    std::vector<size_t> chunk_dims = {1, 1, nprod};

    DataSet tvis = file->createDataSet("vis", vis_space, create_datatype<complex_int>(), chunk_dims);
    vis = std::unique_ptr<DataSet>(new DataSet(tvis));

    file->flush();

}


void visFile::addSample(time_ctype time, std::vector<complex_int> vis) {

    size_t ntime = time_imap->getSpace().getDimensions()[0];

    // Increase the size of the time imap and write the new time entry
    std::cout << "Current size: " << ntime << std::endl << "New size: " << ntime + 1 << std::endl;
    time_imap->resize({ntime + 1});
    time_imap->select({ntime}, {1}).write(&time);

    file->flush();


}


// Add support for all our custom types to HighFive
template <> inline DataType HighFive::create_datatype<freq_ctype>() {
    CompoundType f;
    f.addMember("centre", H5T_IEEE_F64LE);
    f.addMember("width", H5T_IEEE_F64LE);
    f.autoCreate();
    return f;
}

template <> inline DataType HighFive::create_datatype<time_ctype>() {
    CompoundType t;
    t.addMember("fpga_count", H5T_STD_U64LE);
    t.addMember("ctime", H5T_IEEE_F64LE);
    t.autoCreate();
    return t;
}

template <> inline DataType HighFive::create_datatype<input_ctype>() {

    CompoundType i;
    hid_t s32 = H5Tcopy(H5T_C_S1);
    H5Tset_size(s32, 32);
    //AtomicType<char[32]> s32;
    i.addMember("chan_id", H5T_STD_U16LE, 0);
    i.addMember("correlator_input", s32, 2);
    i.manualCreate(34);

    return i;
}

template <> inline DataType HighFive::create_datatype<prod_ctype>() {

    CompoundType p;
    p.addMember("input_a", H5T_STD_U16LE);
    p.addMember("input_b", H5T_STD_U16LE);
    p.autoCreate();
    return p;
}

template <> inline DataType HighFive::create_datatype<complex_int>() {
    CompoundType c;
    c.addMember("r", H5T_STD_I32LE);
    c.addMember("i", H5T_STD_I32LE);
    c.autoCreate();
    return c;
}
