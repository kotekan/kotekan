#include "testDataGenFewHot.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata, CHORD_META_MAX_FREQ
#include "kotekanLogging.hpp"  // for INFO, DEBUG, ERROR

#include <assert.h> // for assert
#include <cmath>    // for fmod
#include <stdint.h> // for int8_t, uint32_t, uint8_t, int16_t, int32_t, uint64_t
#include <string.h> // for memset


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testDataGenFewHot);

#define NUM_ELEMNS 2048
#define NUM_FREQ 256

testDataGenFewHot::testDataGenFewHot(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testDataGenFewHot::main_thread, this)),
    type(config.get<std::string>(unique_name, "type")),
    freq_id(config.get<std::vector<freq_id_t>>(unique_name, "freq_id")),
    elemns(config.get<std::vector<int>>(unique_name, "elemns")),
    samples_per_dataset(config.get<ptrdiff_t>(unique_name, "samples_per_dataset")) {

    buf = get_buffer("out_buf");
    buf->register_producer(unique_name);
    assert(type == "fewhot");

    bool all_el_good = true;
    for (auto const el : elemns) {
        if (!(el < NUM_ELEMNS)) {
            all_el_good = false;
            break;
        }
    }
    if (!all_el_good)
        FATAL_ERROR("Elements {:s} must be in allowed range 0 < el < {:d}",
                    fmt::format("{:s}", fmt::join(elemns, ", ")), NUM_ELEMNS);

    assert(buf->frame_size
           == 1024*256*NUM_ELEMNS * sizeof(kotekan::GetType<kotekan::int4x2_swapped_withoffset>::type));

    //_manual_freq_ids = config.get_default<std::vector<uint32_t>>(unique_name, "manual_freq_ids",
    //                                                             std::vector<uint32_t>());
}


void testDataGenFewHot::main_thread() {

    int seq_num = 0;

    while (!stop_thread) {
        const int frame_id = seq_num % buf->num_frames;
        uint8_t* frame = (uint8_t*)buf->wait_for_empty_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(buf, frame_id);
        assert(chordmeta && "metadata must be of type chordMetadata");

        chordmeta->set_fpga_seq_num(seq_num * samples_per_dataset);
        chordmeta->set_time_downsampling_fpga(1);

        // TODO: Fix this, cannot change from frame to frame (and should not be "now")
        struct timeval now;
        gettimeofday(&now, nullptr);
        chordmeta->set_first_packet_recv_time(now);

        chordmeta->ndishes = 1024;
        chordmeta->n_dish_locations_ew = 4;
        chordmeta->n_dish_locations_ns = 256;
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
        chordmeta->dish_index = const_cast<dish_index_t*>(const_dish_indices);

        chordmeta->set_name("Ebar");
        chordmeta->dims = 4;
        chordmeta->set_array_dimension(0, 1024, "Tbar");
        chordmeta->set_array_dimension(1, 256, "Fbar");
        chordmeta->set_array_dimension(2, 2, "P");
        chordmeta->set_array_dimension(3, NUM_ELEMNS/2, "D");
        chordmeta->set_strides_simple();
        chordmeta->type = kotekan::int4x2_swapped_withoffset;
        std::vector<int> coarse_freq(NUM_FREQ);
        std::vector<int> freq_upchan_factor(coarse_freq.size());
        std::vector<int> freq_upchan_index(coarse_freq.size());
        for(int f = 0 ; f < 256 ; f++) {
          coarse_freq.at(f) = freq_id.at(f/16);
          freq_upchan_factor.at(f) = 16;
          freq_upchan_index.at(f) = f % 16;
        }

        chordmeta->set_coarse_freq(coarse_freq);
        chordmeta->set_freq_upchan_factor(freq_upchan_factor);
        chordmeta->set_freq_upchan_index(freq_upchan_index);

        chordmeta->set_frame_counter(seq_num);

        buf->allocate_ndarray_frame_desc<kotekan::GetType<kotekan::int4x2_swapped_withoffset>::type,
                                         4>("Ebar", {1024,256,2, NUM_ELEMNS/2}, {"Tbar", "Fbar", "P", "D"});
        /* test that things are consistent */
        chordmeta->check_frame_desc(buf->get_ndarray_frame_desc());

        std::memset(frame, 0x88 /* 0 volts */, 1024*256*NUM_ELEMNS);

        for (int i = 0; i < 1024*256; ++i) {
            for (const auto el : elemns) {
                frame[i * NUM_ELEMNS + el] += 0x10;
            }
        }

        buf->mark_frame_full(unique_name, frame_id);

        seq_num += 1;
    }
}
