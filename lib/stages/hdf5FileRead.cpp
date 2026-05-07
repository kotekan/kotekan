#include <Config.hpp>                             // for Config
#include <DataType.hpp>                           // for string_to_type, DataType
#include <Stage.hpp>                              // for Stage
#include <StageFactory.hpp>                       // for REGISTER_KOTEKAN_STAGE
#include <Symbol.hpp>                             // for Symbol
#include <buffer.hpp>                             // for Buffer
#include <bufferContainer.hpp>                    // for bufferContainer
#include <chordMetadata.hpp>                      // for chordMetadata, metadata_is_chord, get_c...
#include <hdf5Files.hpp>                          // for chord_metadata_version
#include <highfive/H5Attribute.hpp>               // for Attribute, Attribute::read
#include <highfive/H5DataSet.hpp>                 // for DataSet, AnnotateTraits::getAttribute
#include <highfive/H5DataSpace.hpp>               // for DataSpace, DataSpace::getDimensions
#include <highfive/H5Exception.hpp>               // for FileException
#include <highfive/H5File.hpp>                    // for File, File::File, NodeTraits::getDataSet
#include <highfive/bits/H5Slice_traits_misc.hpp>  // for SliceTraits::read_raw
#include <kotekanLogging.hpp>                     // for DEBUG, FATAL_ERROR, ERROR
#include <metadata.hpp>                           // for metadataObject
#include <prometheusMetrics.hpp>                  // for Metrics, Gauge
#include <unistd.h>                               // for gethostname, sleep
#include <visUtil.hpp>                            // for current_time
#include <algorithm>                              // for copy
#include <array>                                  // for array
#include <cassert>                                // for assert
#include <cstddef>                                // for ptrdiff_t
#include <cstdint>                                // for int64_t, uint8_t
#include <functional>                             // for function
#include <iomanip>                                // for operator<<, setfill, setw
#include <memory>                                 // for allocator, shared_ptr, __shared_ptr_access
#include <sstream>                                // for basic_ostream, operator<<, basic_ostrin...
#include <string>                                 // for basic_string, char_traits, string, oper...
#include <vector>                                 // for vector

#include "fmt.hpp"                                // for compile_string_to_view

using namespace hdf5;
using namespace HighFive;

class hdf5FileRead : public kotekan::Stage {
    const std::string input_dir = config.get<std::string>(unique_name, "input_dir");
    const std::string file_name = config.get<std::string>(unique_name, "file_name");
    const bool prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    const bool prefix_host_rank = config.get_default<bool>(unique_name, "prefix_host_rank", false);
    const int host_pool_rank = config.get_default<int>(unique_name, "frequency_pool_rank", 0);
    const int host_pool_size = config.get_default<int>(unique_name, "frequency_pool_size", 1);
    const bool do_once = config.get_default<bool>(unique_name, "do_once", false);

    Buffer* const buffer;

public:
    hdf5FileRead(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container) :
        Stage(config, unique_name, buffer_container,
              [](const kotekan::Stage& stage) {
                  return const_cast<kotekan::Stage&>(stage).main_thread();
              }),
        buffer(get_buffer("out_buf")) {
        assert(buffer);
        buffer->register_producer(unique_name);
    }

    virtual ~hdf5FileRead() {}

    void main_thread() override {
        auto& read_time_metric = kotekan::prometheus::Metrics::instance().add_gauge(
            "kotekan_hdf5fileread_read_time_seconds", unique_name);

        for (int frame_index = 0;; ++frame_index) {
            const int frame_id = frame_index % buffer->num_frames;

        wait:

            if (stop_thread)
                break;

            if (do_once && frame_index > 0) {
                sleep(1);
                goto wait;
            }

            // Start timer
            const double t0 = current_time();

            // Define file name
            std::ostringstream buf;
            buf << input_dir << "/";
            if (prefix_hostname) {
                char hostname[256];
                gethostname(hostname, sizeof hostname);
                buf << hostname << "_";
            }
            if (prefix_host_rank) {
                buf << "x" << std::setw(4) << std::setfill('0') << host_pool_rank << "_";
            }
            buf << file_name << "." << std::setw(8) << std::setfill('0') << frame_index << ".h5";
            const std::string full_path = buf.str();

            // Open HDF5 file
            try {
                const File file(full_path, File::ReadOnly);

                // Wait for buffer
                DEBUG("[{:s}/{:d}] Waiting for buffer...", buffer->buffer_name, frame_index);
                std::uint8_t* const frame = buffer->wait_for_empty_frame(unique_name, frame_id);
                if (!frame)
                    break;

                // Read metadata (attributes)
                buffer->allocate_new_metadata_object(frame_id);
                const std::shared_ptr<metadataObject> metadata = buffer->get_metadata(frame_id);
                if (!metadata)
                    FATAL_ERROR("Buffer \"{:s}\" frame {:d} does not have metadata",
                                buffer->buffer_name, frame_id);
                assert(metadata);
                if (!metadata_is_chord(metadata))
                    FATAL_ERROR("Metadata of buffer \"{:s}\" frame {:d} is not of type CHORD",
                                buffer->buffer_name, frame_id);
                assert(metadata_is_chord(metadata));
                const std::shared_ptr<chordMetadata> meta = get_chord_metadata(metadata);
                assert(meta);

                // Open dataset
                const auto dataset = file.getDataSet(file_name);
                const auto space = dataset.getSpace();
                const auto type = dataset.getDataType();
                const auto dims = space.getDimensions();

                {
                    const auto metadata_version =
                        dataset.getAttribute("chord_metadata_version").read<std::array<int, 2>>();
                    const int major = metadata_version[0];
                    const int minor = metadata_version[1];
                    assert(major >= 0);
                    assert(minor >= 0);
                    assert(major == chord_metadata_version.at(0));
                    assert(minor <= chord_metadata_version.at(1));
                }

                meta->set_name(dataset.getAttribute("name").read<std::string>());
                meta->type =
                    kotekan::string_to_type(dataset.getAttribute("type").read<std::string>());
                meta->dims = space.getNumberDimensions();
                assert(meta->dims <= CHORD_META_MAX_DIM);
                const auto dim_names =
                    dataset.getAttribute("dim_names").read<std::vector<std::string>>();
                assert(std::ptrdiff_t(dim_names.size()) == meta->dims);
                for (int d = 0; d < meta->dims; ++d)
                    meta->set_array_dimension(d, dims.at(d), dim_names.at(d));
                {
                    std::ptrdiff_t npoints = 1;
                    for (int d = meta->dims - 1; d >= 0; --d) {
                        meta->stride[d] = npoints;
                        npoints *= meta->dim[d];
                    }
                    assert(std::ptrdiff_t(space.getElementCount()) == npoints);
                }
                meta->offset = 0;

                if (dataset.hasAttribute("fpga_seq_num"))
                    meta->set_fpga_seq_num(
                        dataset.getAttribute("fpga_seq_num").read<std::int64_t>());
                if (dataset.hasAttribute("time_downsampling_fpga"))
                    meta->set_time_downsampling_fpga(
                        dataset.getAttribute("time_downsampling_fpga").read<int>());

                if (dataset.hasAttribute("coarse_freq"))
                    meta->set_coarse_freq(
                        dataset.getAttribute("coarse_freq").read<std::vector<int>>());
                if (dataset.hasAttribute("freq_upchan_factor"))
                    meta->set_freq_upchan_factor(
                        dataset.getAttribute("freq_upchan_factor").read<std::vector<int>>());
                if (dataset.hasAttribute("freq_upchan_index"))
                    meta->set_freq_upchan_index(
                        dataset.getAttribute("freq_upchan_index").read<std::vector<int>>());

                if (dataset.hasAttribute("rfi_frame_excision_enabled"))
                    meta->set_rfi_frame_excision_enabled(
                        dataset.getAttribute("rfi_frame_excision_enabled").read<bool>());
                if (dataset.hasAttribute("rfi_frame_excision_thresholds"))
                    meta->set_rfi_frame_excision_thresholds(
                        dataset.getAttribute("rfi_frame_excision_thresholds")
                            .read<std::vector<std::array<float, 2>>>());

                // TODO: Read telescope fields and WARN if they don't match?

                    meta->n_dish_locations_ns =
                        dataset.getAttribute("n_dish_locations_ns").read<int>();
                    meta->n_dish_locations_ew =
                        dataset.getAttribute("n_dish_locations_ew").read<int>();

                    const auto dish_index =
                        dataset.getAttribute("dish_index").read<std::vector<int>>();
                    assert(std::ptrdiff_t(dish_index.size())
                           == meta->n_dish_locations_ns * meta->n_dish_locations_ew);
                    meta->dish_index =
                        new dish_index_t[meta->n_dish_locations_ns * meta->n_dish_locations_ew];
                    std::copy(dish_index.begin(), dish_index.end(), meta->dish_index);

                } else {
                    meta->ndishes = -1;
                    meta->dish_index = nullptr;
                }
        // HACK for now. DONOTPUSH
        meta->ndishes = 1024;
        meta->n_dish_locations_ew = 4;
        meta->n_dish_locations_ns = 256;
        const static dish_index_t const_dish_indices[1024] = {
            0, 256, 512, 768,
            1, 257, 513, 769,
            2, 258, 514, 770,
            3, 259, 515, 771,
            4, 260, 516, 772,
            5, 261, 517, 773,
            6, 262, 518, 774,
            7, 263, 519, 775,
            8, 264, 520, 776,
            9, 265, 521, 777,
            10, 266, 522, 778,
            11, 267, 523, 779,
            12, 268, 524, 780,
            13, 269, 525, 781,
            14, 270, 526, 782,
            15, 271, 527, 783,
            16, 272, 528, 784,
            17, 273, 529, 785,
            18, 274, 530, 786,
            19, 275, 531, 787,
            20, 276, 532, 788,
            21, 277, 533, 789,
            22, 278, 534, 790,
            23, 279, 535, 791,
            24, 280, 536, 792,
            25, 281, 537, 793,
            26, 282, 538, 794,
            27, 283, 539, 795,
            28, 284, 540, 796,
            29, 285, 541, 797,
            30, 286, 542, 798,
            31, 287, 543, 799,
            32, 288, 544, 800,
            33, 289, 545, 801,
            34, 290, 546, 802,
            35, 291, 547, 803,
            36, 292, 548, 804,
            37, 293, 549, 805,
            38, 294, 550, 806,
            39, 295, 551, 807,
            40, 296, 552, 808,
            41, 297, 553, 809,
            42, 298, 554, 810,
            43, 299, 555, 811,
            44, 300, 556, 812,
            45, 301, 557, 813,
            46, 302, 558, 814,
            47, 303, 559, 815,
            48, 304, 560, 816,
            49, 305, 561, 817,
            50, 306, 562, 818,
            51, 307, 563, 819,
            52, 308, 564, 820,
            53, 309, 565, 821,
            54, 310, 566, 822,
            55, 311, 567, 823,
            56, 312, 568, 824,
            57, 313, 569, 825,
            58, 314, 570, 826,
            59, 315, 571, 827,
            60, 316, 572, 828,
            61, 317, 573, 829,
            62, 318, 574, 830,
            63, 319, 575, 831,
            64, 320, 576, 832,
            65, 321, 577, 833,
            66, 322, 578, 834,
            67, 323, 579, 835,
            68, 324, 580, 836,
            69, 325, 581, 837,
            70, 326, 582, 838,
            71, 327, 583, 839,
            72, 328, 584, 840,
            73, 329, 585, 841,
            74, 330, 586, 842,
            75, 331, 587, 843,
            76, 332, 588, 844,
            77, 333, 589, 845,
            78, 334, 590, 846,
            79, 335, 591, 847,
            80, 336, 592, 848,
            81, 337, 593, 849,
            82, 338, 594, 850,
            83, 339, 595, 851,
            84, 340, 596, 852,
            85, 341, 597, 853,
            86, 342, 598, 854,
            87, 343, 599, 855,
            88, 344, 600, 856,
            89, 345, 601, 857,
            90, 346, 602, 858,
            91, 347, 603, 859,
            92, 348, 604, 860,
            93, 349, 605, 861,
            94, 350, 606, 862,
            95, 351, 607, 863,
            96, 352, 608, 864,
            97, 353, 609, 865,
            98, 354, 610, 866,
            99, 355, 611, 867,
            100, 356, 612, 868,
            101, 357, 613, 869,
            102, 358, 614, 870,
            103, 359, 615, 871,
            104, 360, 616, 872,
            105, 361, 617, 873,
            106, 362, 618, 874,
            107, 363, 619, 875,
            108, 364, 620, 876,
            109, 365, 621, 877,
            110, 366, 622, 878,
            111, 367, 623, 879,
            112, 368, 624, 880,
            113, 369, 625, 881,
            114, 370, 626, 882,
            115, 371, 627, 883,
            116, 372, 628, 884,
            117, 373, 629, 885,
            118, 374, 630, 886,
            119, 375, 631, 887,
            120, 376, 632, 888,
            121, 377, 633, 889,
            122, 378, 634, 890,
            123, 379, 635, 891,
            124, 380, 636, 892,
            125, 381, 637, 893,
            126, 382, 638, 894,
            127, 383, 639, 895,
            128, 384, 640, 896,
            129, 385, 641, 897,
            130, 386, 642, 898,
            131, 387, 643, 899,
            132, 388, 644, 900,
            133, 389, 645, 901,
            134, 390, 646, 902,
            135, 391, 647, 903,
            136, 392, 648, 904,
            137, 393, 649, 905,
            138, 394, 650, 906,
            139, 395, 651, 907,
            140, 396, 652, 908,
            141, 397, 653, 909,
            142, 398, 654, 910,
            143, 399, 655, 911,
            144, 400, 656, 912,
            145, 401, 657, 913,
            146, 402, 658, 914,
            147, 403, 659, 915,
            148, 404, 660, 916,
            149, 405, 661, 917,
            150, 406, 662, 918,
            151, 407, 663, 919,
            152, 408, 664, 920,
            153, 409, 665, 921,
            154, 410, 666, 922,
            155, 411, 667, 923,
            156, 412, 668, 924,
            157, 413, 669, 925,
            158, 414, 670, 926,
            159, 415, 671, 927,
            160, 416, 672, 928,
            161, 417, 673, 929,
            162, 418, 674, 930,
            163, 419, 675, 931,
            164, 420, 676, 932,
            165, 421, 677, 933,
            166, 422, 678, 934,
            167, 423, 679, 935,
            168, 424, 680, 936,
            169, 425, 681, 937,
            170, 426, 682, 938,
            171, 427, 683, 939,
            172, 428, 684, 940,
            173, 429, 685, 941,
            174, 430, 686, 942,
            175, 431, 687, 943,
            176, 432, 688, 944,
            177, 433, 689, 945,
            178, 434, 690, 946,
            179, 435, 691, 947,
            180, 436, 692, 948,
            181, 437, 693, 949,
            182, 438, 694, 950,
            183, 439, 695, 951,
            184, 440, 696, 952,
            185, 441, 697, 953,
            186, 442, 698, 954,
            187, 443, 699, 955,
            188, 444, 700, 956,
            189, 445, 701, 957,
            190, 446, 702, 958,
            191, 447, 703, 959,
            192, 448, 704, 960,
            193, 449, 705, 961,
            194, 450, 706, 962,
            195, 451, 707, 963,
            196, 452, 708, 964,
            197, 453, 709, 965,
            198, 454, 710, 966,
            199, 455, 711, 967,
            200, 456, 712, 968,
            201, 457, 713, 969,
            202, 458, 714, 970,
            203, 459, 715, 971,
            204, 460, 716, 972,
            205, 461, 717, 973,
            206, 462, 718, 974,
            207, 463, 719, 975,
            208, 464, 720, 976,
            209, 465, 721, 977,
            210, 466, 722, 978,
            211, 467, 723, 979,
            212, 468, 724, 980,
            213, 469, 725, 981,
            214, 470, 726, 982,
            215, 471, 727, 983,
            216, 472, 728, 984,
            217, 473, 729, 985,
            218, 474, 730, 986,
            219, 475, 731, 987,
            220, 476, 732, 988,
            221, 477, 733, 989,
            222, 478, 734, 990,
            223, 479, 735, 991,
            224, 480, 736, 992,
            225, 481, 737, 993,
            226, 482, 738, 994,
            227, 483, 739, 995,
            228, 484, 740, 996,
            229, 485, 741, 997,
            230, 486, 742, 998,
            231, 487, 743, 999,
            232, 488, 744, 1000,
            233, 489, 745, 1001,
            234, 490, 746, 1002,
            235, 491, 747, 1003,
            236, 492, 748, 1004,
            237, 493, 749, 1005,
            238, 494, 750, 1006,
            239, 495, 751, 1007,
            240, 496, 752, 1008,
            241, 497, 753, 1009,
            242, 498, 754, 1010,
            243, 499, 755, 1011,
            244, 500, 756, 1012,
            245, 501, 757, 1013,
            246, 502, 758, 1014,
            247, 503, 759, 1015,
            248, 504, 760, 1016,
            249, 505, 761, 1017,
            250, 506, 762, 1018,
            251, 507, 763, 1019,
            252, 508, 764, 1020,
            253, 509, 765, 1021,
            254, 510, 766, 1022,
            255, 511, 767, 1023,
        };
        meta->dish_index = const_cast<dish_index_t*>(const_dish_indices);

                {
                    /* new style array description */
                    const kotekan::DataType value_type =
                        kotekan::string_to_type(dataset.getAttribute("type").read<std::string>());
                    assert(value_type != kotekan::unknown_type);
                    const std::string name = dataset.getAttribute("name").read<std::string>();

                    std::vector<ptrdiff_t> dimensions(dims.begin(), dims.end());
                    std::vector<kotekan::Symbol> dimnames(dim_names.begin(), dim_names.end());

                    buffer->allocate_ndarray_frame_desc(value_type, name, dimensions, dimnames);
                    /* test that things are consistent */
                    meta->check_frame_desc(buffer->get_ndarray_frame_desc());
                }

                // Read buffer
                DEBUG("[{:s}/{:d}] Filling buffer...", buffer->buffer_name, frame_index);
                dataset.read_raw(frame, type);

                // Mark buffer as full
                DEBUG("[{:s}/{:d}] Marking buffer as full...", buffer->buffer_name, frame_index);
                buffer->mark_frame_full(unique_name, frame_id);

                // Stop timer
                const double t1 = current_time();
                const double elapsed = t1 - t0;
                read_time_metric.set(elapsed);
            } catch (const FileException& ex) {
                if (frame_index == 0)
                    FATAL_ERROR("Could not open HDF5 file {:s}: {:s}", full_path, ex.what());
                else
                    ERROR("Could not open HDF5 file {:s}, terminating reader", full_path);
                break;
            }

        } // while !stop_thread
    }
};

REGISTER_KOTEKAN_STAGE(hdf5FileRead);
