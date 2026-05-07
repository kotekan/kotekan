#include "chordMetadata.hpp"

#include "Symbol.hpp"  // for Symbol
#include "factory.hpp" // for REGISTER_TYPE_WITH_FACTORY

#include <algorithm>   // for copy_n, copy, max
#include <cstring>     // for memset
#include <string.h>    // for strncmp, strncpy, memset, strnlen
#include <type_traits> // for is_pod_v

REGISTER_TYPE_WITH_FACTORY(metadataObject, chordMetadata);

chordMetadata::chordMetadata() :
    type(kotekan::unknown_type), dims(-1), offset(0), ndishes(-1), n_dish_locations_ew(-1),
    n_dish_locations_ns(-1), dish_index(nullptr) {
    name[0] = '\0';
    for (int d = 0; d < CHORD_META_MAX_DIM; ++d) {
        dim[d] = -1;
        dim_name[d][0] = '\0';
        stride[d] = -1;
    }
}

bool chordMetadata::operator==(const chordMetadata& other) const {
    if (this == &other)
        return true;

    std::scoped_lock<std::mutex, std::mutex> lock(this->lock, other.lock);

    if (0 != strncmp(name, other.name, CHORD_META_MAX_DIMNAME))
        return false;

    if (type != other.type)
        return false;

    if (dims != other.dims)
        return false;

    for (int d = 0; d < dims; ++d)
        if (dim[d] != other.dim[d])
            return false;
    for (int d = 0; d < dims; ++d)
        if (0 != strncmp(dim_name[d], other.dim_name[d], CHORD_META_MAX_DIMNAME))
            return false;
    for (int d = 0; d < dims; ++d)
        if (stride[d] != other.stride[d])
            return false;
    if (offset != other.offset)
        return false;

    // TODO: this misses dish_positions etc
    return metadata == other.metadata;
}

void chordMetadata::check_frame_desc(
    const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) const {

    bool failed = false;

    if (strncmp(this->name, frame_desc->get_quantity_name().get_c_string(), sizeof(this->name))
        != 0) {
        ERROR("Names differ: {:s} != {:s}",
              std::string(this->name, strnlen(this->name, sizeof(this->name))),
              frame_desc->get_quantity_name());
        failed = true;
    }
    if (this->type != frame_desc->get_value_datatype()) {
        ERROR("Types differ for {:s}: {:s} != {:s}", frame_desc->get_quantity_name(),
              kotekan::type_to_string(this->type),
              kotekan::type_to_string(frame_desc->get_value_datatype()));
        failed = true;
    }
    if (size_t(this->dims) != frame_desc->get_rank()) {
        ERROR("Ranks differ for {:s}: {:d} != {:d}", frame_desc->get_quantity_name(), this->dims,
              frame_desc->get_rank());
        failed = true;
    }
    for (int d = this->dims - 1; d >= 0; --d) {

        if (strncmp(this->dim_name[d], frame_desc->get_dimname(d).get_c_string(),
                    sizeof this->dim_name[d])
            != 0) {
            ERROR("Dim_name[{:d}] differs for {:s}: {:s} != {:s}", d,
                  frame_desc->get_quantity_name(),
                  std::string(this->dim_name[d],
                              strnlen(this->dim_name[d], sizeof(this->dim_name[d]))),
                  frame_desc->get_dimname(d));
            failed = true;
        }

        if (this->dim[d] != frame_desc->get_extent(d)) {
            ERROR("Dim[{:d}] differs for {:s}: {:d} != {:d}", d, frame_desc->get_quantity_name(),
                  this->dim[d], frame_desc->get_extent(d));
            failed = true;
        }
        if (this->stride[d] != frame_desc->get_stride(d)) {
            ERROR("Stride[{:d}] differs for {:s}: {:d} != {:d}", d, frame_desc->get_quantity_name(),
                  this->stride[d], frame_desc->get_stride(d));
            failed = true;
        }
    }

    if (failed)
        FATAL_ERROR("Inconsistent array description between CHORDMetadata and FrameDesc");
}

void chordMetadata::set_from_frame_desc(
    const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) {
    set_name(frame_desc->get_quantity_name());
    this->type = frame_desc->get_value_datatype();
    this->dims = frame_desc->get_rank();
    for (int d = this->dims - 1; d >= 0; --d) {
        set_array_dimension(d, frame_desc->get_extent(d), frame_desc->get_dimname(d));
        this->stride[d] = frame_desc->get_stride(d);
    }
    // HACK for now. DONOTPUSH
    this->ndishes = 1024;
    this->n_dish_locations_ew = 4;
    this->n_dish_locations_ns = 256;
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
    this->dish_index = const_cast<dish_index_t*>(const_dish_indices);
}

void chordMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto chord_other = std::dynamic_pointer_cast<const chordMetadata>(other);
    assert(chord_other);

    if (this == chord_other.get())
        return;

    // order locks so that there is no race condition if two chordMetadata a and
    // b are deepCopy'ed into each other a the same time
    std::scoped_lock<std::mutex, std::mutex> lock(this->lock, chord_other->lock);
    *this = *chord_other;
}

struct chordMetadataFormat {
    int32_t max_dim;
    int32_t max_dimname;
    int32_t max_freq;

    int32_t frame_counter;
    int64_t fpga_seq_num;

    // Time sampling -- the factor by which the time samples have been
    // downsampled relative to FPGA samples.
    int32_t time_downsampling_fpga;

    char name[CHORD_META_MAX_DIMNAME]; // "E", "J", "I", etc
    // chordDataType type;
    int32_t type;

    int32_t dims;
    int32_t dim[CHORD_META_MAX_DIM];
    char dim_name[CHORD_META_MAX_DIM][CHORD_META_MAX_DIMNAME]; // "F", "Tbar", "D", etc
    int64_t stride[CHORD_META_MAX_DIM];
    int64_t offset;

    // Per-frequency arrays
    int32_t nfreq;

    // frequencies -- integer (0-2047) identifier for FPGA coarse frequencies
    int32_t coarse_freq[CHORD_META_MAX_FREQ];

    // the upchannelization factor that each frequency has gone through (1 for = FPGA)
    int32_t freq_upchan_factor[CHORD_META_MAX_FREQ];

    // the upchannelization index for each frequency (0 ... freq_upchan_factor - 1)
    int32_t freq_upchan_index[CHORD_META_MAX_FREQ];

    uint32_t rfi_num_bad_inputs;
    int32_t rfi_flagged_samples;
    int32_t lost_timesamples;

    chordMetadata::beamCoord beam_coord;
    static_assert(std::is_pod_v<chordMetadata::beamCoord> == true,
                  "beamCoord containes C++ only data");

    stream_t stream_id;
    static_assert(std::is_pod_v<stream_t> == true, "stream_t containes C++ only data");

    // cannot use dset_id_t here since Hash is not a pod (has a constructor)
    char dataset_id[32];

    timeval first_packet_recv_time;
};

size_t chordMetadata::get_serialized_size() {
    return sizeof(chordMetadataFormat);
}

size_t chordMetadata::set_from_bytes(const char* bytes, [[maybe_unused]] size_t length) {
    (void)length;
    assert(length >= get_serialized_size());
    assert(length >= sizeof(chordMetadataFormat));

    const chordMetadataFormat* fmt = reinterpret_cast<const chordMetadataFormat*>(bytes);

    if (fmt->frame_counter != -1)
        this->set_frame_counter(fmt->frame_counter);
    if (fmt->fpga_seq_num != -1)
        this->set_fpga_seq_num(fmt->fpga_seq_num);
    if (fmt->time_downsampling_fpga != -1)
        this->set_time_downsampling_fpga(fmt->time_downsampling_fpga);
    for (int i = 0; i < CHORD_META_MAX_DIMNAME; i++) {
        name[i] = fmt->name[i];
    }
    type = (kotekan::DataType)fmt->type;
    assert(CHORD_META_MAX_DIM == fmt->max_dim);
    assert(CHORD_META_MAX_DIMNAME == fmt->max_dimname);
    assert(CHORD_META_MAX_FREQ == fmt->max_freq);
    dims = fmt->dims;
    assert(dims < CHORD_META_MAX_DIM);
    for (int i = 0; i < dims; i++) {
        dim[i] = fmt->dim[i];
        for (int j = 0; j < CHORD_META_MAX_DIMNAME; j++) {
            dim_name[i][j] = fmt->dim_name[i][j];
        }
        stride[i] = fmt->stride[i];
    }
    offset = fmt->offset;
    const int nfreq = fmt->nfreq;
    assert(nfreq < CHORD_META_MAX_FREQ);
    if (fmt->freq_upchan_factor[0] != 0) // 0 should be an invalid value
        this->set_freq_upchan_factor(
            std::vector<int>(fmt->freq_upchan_factor, fmt->freq_upchan_factor + nfreq));
    if (fmt->freq_upchan_index[0] != -1) // -1 should be an invalid value
        this->set_freq_upchan_index(
            std::vector<int>(fmt->freq_upchan_index, fmt->freq_upchan_index + nfreq));
    if (fmt->coarse_freq[0] != -1) // -1 is an invalid frequency index
        this->set_coarse_freq(std::vector<int>(fmt->coarse_freq, fmt->coarse_freq + nfreq));

    if (fmt->rfi_num_bad_inputs != uint32_t(-1))
        this->set_rfi_num_bad_inputs(fmt->rfi_num_bad_inputs);
    if (fmt->rfi_flagged_samples != -1)
        this->set_rfi_flagged_samples(fmt->rfi_flagged_samples);
    if (fmt->lost_timesamples != -1)
        this->set_lost_timesamples(fmt->lost_timesamples);

    if (fmt->stream_id.id != uint64_t(-1))
        this->set_stream_id(fmt->stream_id);
    if (fmt->dataset_id[0] != '\0')
        this->set_dataset_id(
            dset_id_t::from_string(std::string(fmt->dataset_id, sizeof(fmt->dataset_id))));

    if (fmt->beam_coord.scaling[0] != 0)
        this->set_beam_coord(fmt->beam_coord);

    if (fmt->first_packet_recv_time.tv_sec != 0) // sometime in 1970
        this->set_first_packet_recv_time(fmt->first_packet_recv_time);

    // TODO: this misses dish_positions etc
    return sizeof(chordMetadataFormat);
}

size_t chordMetadata::serialize(char* bytes) {
    chordMetadataFormat* fmt = reinterpret_cast<chordMetadataFormat*>(bytes);
    memset(fmt, 0, sizeof(chordMetadataFormat));

    fmt->max_dim = CHORD_META_MAX_DIM;
    fmt->max_dimname = CHORD_META_MAX_DIMNAME;
    fmt->max_freq = CHORD_META_MAX_FREQ;

    if (this->has_frame_counter())
        fmt->frame_counter = this->get_frame_counter();
    else
        fmt->frame_counter = -1;
    if (this->has_fpga_seq_num())
        fmt->fpga_seq_num = this->get_fpga_seq_num();
    else
        fmt->fpga_seq_num = -1;
    if (this->has_time_downsampling_fpga())
        fmt->time_downsampling_fpga = this->get_time_downsampling_fpga();
    else
        fmt->time_downsampling_fpga = -1;
    for (int i = 0; i < CHORD_META_MAX_DIMNAME; i++) {
        fmt->name[i] = name[i];
    }
    fmt->type = (int32_t)type;
    fmt->dims = dims;
    for (int i = 0; i < dims; i++) {
        fmt->dim[i] = dim[i];
        for (int j = 0; j < CHORD_META_MAX_DIMNAME; j++) {
            fmt->dim_name[i][j] = dim_name[i][j];
        }
        fmt->stride[i] = stride[i];
    }
    fmt->offset = offset;
    fmt->nfreq = this->get_nfreq();
    assert(fmt->nfreq < CHORD_META_MAX_FREQ);
    if (this->has_freq_upchan_factor())
        std::copy_n(this->get_freq_upchan_factor().data(), this->get_nfreq(),
                    fmt->freq_upchan_factor);
    if (this->has_freq_upchan_index())
        std::copy_n(this->get_freq_upchan_index().data(), this->get_nfreq(),
                    fmt->freq_upchan_index);
    else
        std::memset(
            fmt->freq_upchan_index, -1,
            this->get_nfreq()
                * sizeof(fmt->freq_upchan_index[0])); // set bytes not ints but is ok for now
    if (this->has_coarse_freq())
        std::copy_n(this->get_coarse_freq().data(), this->get_nfreq(), fmt->coarse_freq);
    else
        std::memset(fmt->coarse_freq, -1,
                    this->get_nfreq()
                        * sizeof(fmt->coarse_freq[0])); // set bytes not ints but is ok for now

    if (this->has_rfi_num_bad_inputs())
        fmt->rfi_num_bad_inputs = this->get_rfi_num_bad_inputs();
    else
        fmt->rfi_num_bad_inputs = uint32_t(-1);
    if (this->has_rfi_flagged_samples())
        fmt->rfi_flagged_samples = this->get_rfi_flagged_samples();
    else
        fmt->rfi_flagged_samples = -1;
    if (this->has_lost_timesamples())
        fmt->lost_timesamples = this->get_lost_timesamples();
    else
        fmt->lost_timesamples = -1;

    if (this->has_stream_id())
        fmt->stream_id = this->get_stream_id();
    else
        fmt->stream_id = stream_t{uint64_t(-1)};
    if (this->has_dataset_id()) {
        const std::string dataset_id_str = this->get_dataset_id().to_string();
        assert(dataset_id_str.size() == sizeof(fmt->dataset_id)
               && "Sized of strigified hash is not 32");
        std::copy_n(dataset_id_str.data(), sizeof(fmt->dataset_id), fmt->dataset_id);
    }

    if (this->has_beam_coord())
        fmt->beam_coord = this->get_beam_coord();

    if (this->has_first_packet_recv_time())
        fmt->first_packet_recv_time = this->get_first_packet_recv_time();
    else
        fmt->first_packet_recv_time = timeval{0, 0}; // unix epoch, 1970

    // TODO: this misses dish_positions etc
    return sizeof(chordMetadataFormat);
}

nlohmann::json chordMetadata::to_json() {
    nlohmann::json rtn = {};
    ::to_json(rtn, *this);
    return rtn;
}

void to_json(nlohmann::json& j, const chordMetadata& m) {
    assert(j.empty());

    j = m.metadata;

    j.emplace("max_dim", CHORD_META_MAX_DIM);
    j.emplace("max_dimname", CHORD_META_MAX_DIMNAME);
    j.emplace("max_freq", CHORD_META_MAX_FREQ);

    j.emplace("name", std::string(m.name, strnlen(m.name, sizeof(m.name))));
    j.emplace("type", m.type);
    j.emplace("dims", m.dims);
    j.emplace("dim", std::vector<int>(m.dim, m.dim + m.dims));
    std::vector<std::string> dimnames;
    for (int i = 0; i < m.dims; i++)
        dimnames.push_back(
            std::string(m.dim_name[i], strnlen(m.dim_name[i], sizeof(m.dim_name[i]))));
    j.emplace("dim_name", dimnames);
    j.emplace("stride", std::vector<int64_t>(m.stride, m.stride + m.dims));
    j.emplace("offset", m.offset);
    // TODO: this misses dish_positions etc
}

void from_json(const nlohmann::json& j, chordMetadata& m) {
    // TODO once everything is stored in json, can just copy in the json given
    assert(m.metadata.empty());

    if (j.contains(jsonMetadata::BEAM_COORD))
        m.metadata.emplace(jsonMetadata::BEAM_COORD, j.at(jsonMetadata::BEAM_COORD));
    if (j.contains(jsonMetadata::FPGA_SEQ_NUM))
        m.metadata.emplace(jsonMetadata::FPGA_SEQ_NUM, j.at(jsonMetadata::FPGA_SEQ_NUM));
    if (j.contains(jsonMetadata::TIME_DOWNSAMPLING_FPGA))
        m.metadata.emplace(jsonMetadata::TIME_DOWNSAMPLING_FPGA,
                           j.at(jsonMetadata::TIME_DOWNSAMPLING_FPGA));
    if (j.contains(jsonMetadata::COARSE_FREQ))
        m.metadata.emplace(jsonMetadata::COARSE_FREQ, j.at(jsonMetadata::COARSE_FREQ));
    if (j.contains(jsonMetadata::DATASET_ID))
        m.metadata.emplace(jsonMetadata::DATASET_ID, j.at(jsonMetadata::DATASET_ID));
    if (j.contains(jsonMetadata::RFI_NUM_BAD_INPUTS))
        m.metadata.emplace(jsonMetadata::RFI_NUM_BAD_INPUTS,
                           j.at(jsonMetadata::RFI_NUM_BAD_INPUTS));
    if (j.contains(jsonMetadata::RFI_FLAGGED_SAMPLES))
        m.metadata.emplace(jsonMetadata::RFI_FLAGGED_SAMPLES,
                           j.at(jsonMetadata::RFI_FLAGGED_SAMPLES));
    if (j.contains(jsonMetadata::LOST_TIMESAMPLES))
        m.metadata.emplace(jsonMetadata::LOST_TIMESAMPLES, j.at(jsonMetadata::LOST_TIMESAMPLES));
    if (j.contains(jsonMetadata::STREAM_ID))
        m.metadata.emplace(jsonMetadata::STREAM_ID, j.at(jsonMetadata::STREAM_ID));
    if (j.contains(jsonMetadata::FRAME_COUNTER))
        m.metadata.emplace(jsonMetadata::FRAME_COUNTER, j.at(jsonMetadata::FRAME_COUNTER));

    if (j.contains(jsonMetadata::FIRST_PACKET_RECV_TIME))
        m.metadata.emplace(jsonMetadata::FIRST_PACKET_RECV_TIME,
                           j.at(jsonMetadata::FIRST_PACKET_RECV_TIME));

    if (j.contains(jsonMetadata::FREQ_UPCHAN_FACTOR))
        m.metadata.emplace(jsonMetadata::FREQ_UPCHAN_FACTOR,
                           j.at(jsonMetadata::FREQ_UPCHAN_FACTOR));
    if (j.contains(jsonMetadata::FREQ_UPCHAN_INDEX))
        m.metadata.emplace(jsonMetadata::FREQ_UPCHAN_INDEX, j.at(jsonMetadata::FREQ_UPCHAN_INDEX));

    assert(j.at("max_dim") == CHORD_META_MAX_DIM);
    assert(j.at("max_dimname") == CHORD_META_MAX_DIMNAME);
    assert(j.at("max_freq") == CHORD_META_MAX_FREQ);

    // GCC helpfully tries to warn us that the destination string may end up not
    // NUL-terminated, which we know.
#pragma GCC diagnostic push
#if GCC_VERSION > 80000
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
    strncpy(m.name, j.at("name").template get<std::string>().c_str(), sizeof(m.name));
#pragma GCC diagnostic pop
    m.type = j.at("type").template get<kotekan::DataType>();
    m.dims = j.at("dims").template get<int>();
    std::vector<int> extents = j.at("dim").template get<std::vector<int>>();
    std::copy(extents.begin(), extents.end(), m.dim);
    std::vector<std::string> dimnames = j.at("dim_name").template get<std::vector<std::string>>();
    for (int i = 0; i < m.dims; i++)
        strncpy(m.dim_name[i], dimnames.at(i).c_str(), sizeof(m.dim_name[i]));
    std::vector<int64_t> strides = j.at("stride").template get<std::vector<int64_t>>();
    for (int i = 0; i < m.dims; i++)
        m.stride[i] = strides.at(i);
    m.offset = j.at("offset");
    // TODO: this misses dish_positions etc
}
