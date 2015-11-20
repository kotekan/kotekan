#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <getopt.h>
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

extern "C" {
#include <pthread.h>
}

// DPDK!
extern "C" {
#include <rte_config.h>
#include <rte_common.h>
#include <rte_log.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_pci.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
}

#include "errors.h"
#include "buffers.h"
#include "gpu_thread.h"
#include "network_dna.h"
#include "error_correction.h"
#include "ch_acq_uplink.h"
#include "config.h"
#include "gpu_post_process.h"
#include "beamforming.h"
#include "beamforming_post_process.h"
#include "vdif_stream.h"
#include "null.h"
#include "util.h"
#include "network_dpdk.h"

void print_help() {
    printf("usage: kotekan [opts]\n\n");
    printf("Options:\n");
    printf("    --config (-c) [file]            The local JSON config file to use\n");
    printf("    --log-to-screen (-s)            Output log messages to stderr\n");
    printf("    --log-level (-l) [level]        0 = errors only, 1 = +warnings, 2 = +info, 3 = +debug\n\n");
    printf("    --daemon -d [root]              Run in daemon mode, disables output to stderr");
    printf("Testing Options:\n");
    printf("    --no-network-test (-t)          Generates fake data for testing.\n");
    printf("    --read-file (-f) [file or dir]  Read a packet dump file or dir instead of using the network.\n");
}

void daemonize(char * root_dir, int log_options) {
    // Process ID and Session ID.
    pid_t pid, sid;

    // Fork off the parent process.
    pid = fork();
    if (pid < 0) {
        exit(EXIT_FAILURE);
    }

    // Exit the parrent proces.
    if (pid > 0) {
        FILE * pid_fd = fopen("/var/run/kotekan.pid", "w");
        if (!pid_fd) {
            perror("failed to open PID file");
            exit(EXIT_FAILURE);
        }
        fprintf(pid_fd, "%d\n", pid);
        fclose(pid_fd);
        exit(EXIT_SUCCESS);
    }

    // Change the file mode mask.
    umask(0);

    // Open logs.
    openlog ("kotekan", log_options, LOG_LOCAL1);

    // Create a new SID for the child process
    sid = setsid();
    if (sid < 0) {
        ERROR("Cannot set SID!");
        exit(EXIT_FAILURE);
    }

    // Change the current working directory.
    if ((chdir(root_dir)) < 0) {
        ERROR("Cannot change directory to %s", root_dir);
        exit(EXIT_FAILURE);
    }

    // Close standard file descriptors
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    // TODO Handle signals.
}

void dpdk_setup() {

    char  arg0[] = "./kotekan";
    char  arg1[] = "-n";
    char  arg2[] = "4";
    char  arg3[] = "-c";
    char  arg4[] = "f";
    char* argv2[] = { &arg0[0], &arg1[0], &arg2[0], &arg3[0], &arg4[0], NULL };
    int   argc2   = (int)(sizeof(argv2) / sizeof(argv2[0])) - 1;

    /* Initialize the Environment Abstraction Layer (EAL). */
    int ret2 = rte_eal_init(argc2, argv2);
    if (ret2 < 0)
        exit(EXIT_FAILURE);

    // Can we see ports here?
    int nb_ports = rte_eth_dev_count();
    fprintf(stdout, "kotekan; Number of ports: %d\n", nb_ports);

}

int main(int argc, char ** argv) {

    dpdk_setup();

    int use_ch_acq = 1;
    int read_file = 0;
    char * input_file_name = NULL;
    int opt_val = 0;
    int no_network_test = 0;
    char * config_file_name = (char *)"kotekan.conf";
    struct Config config;
    int error = 0;
    int log_options = LOG_CONS | LOG_PID | LOG_NDELAY;
    int make_daemon = 0;
    char * kotekan_root_dir = (char *)"./";

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int j = 4; j < 12; j++)
        CPU_SET(j, &cpuset);

    for (;;) {
        static struct option long_options[] = {
            {"no-network-test", no_argument, 0, 't'},
            {"read-file", required_argument, 0, 'f'},
            {"config", required_argument, 0, 'c'},
            {"write-local", no_argument, 0, 'w'},
            {"log-level", required_argument, 0, 'l'},
            {"daemon", required_argument, 0, 'd'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        int option_index = 0;
        int option_value = 0;

        opt_val = getopt_long (argc, argv, "htwsf:c:l:d:",
                               long_options, &option_index);

        // End of args
        if (opt_val == -1) {
            break;
        }

        switch (opt_val) {
            case 'h':
                print_help();
                return 0;
                break;
            case 't':
                no_network_test = 1;
                break;
            case 's':
                log_options = log_options | LOG_PERROR;
                break;
            case 'f':
                read_file = 1;
                input_file_name = strdup(optarg);
                break;
            case 'c':
                config_file_name = strdup(optarg);
                break;
            case 'd':
                make_daemon = 1;
                kotekan_root_dir = strdup(optarg);
                break;
            case 'l':
                option_value = atoi(optarg);
                switch (option_value) {
                    case 3:
                        log_level_debug = 1;
                    case 2:
                        log_level_info = 1;
                    case 1:
                        log_level_warn = 1;
                    default:
                        break;
                }
                break;
            default:
                printf("Invalid option, run with -h to see options");
                return -1;
                break;
        }
    }

    if (make_daemon) {
        // Do not use LOG_CONS or LOG_PERROR, since stderr does not exist in daemon mode.
        daemonize(kotekan_root_dir, LOG_PID | LOG_NDELAY);
    } else {
        // Setup syslog
        openlog ("kotekan", log_options, LOG_LOCAL1);
    }

    INFO("kotekan starting..."); /// TODO Include git commit id.

    // Load configuration file.
    error = load_config_from_file(&config, config_file_name);
    if (error) {
        ERROR("Error processing config file, exiting...");
        return -1;
    }

    INFO("Parsed config file %s:", config_file_name);
    //print_config(&config);

    // If we are reading from a file, force the number of links to be 1.
    if (read_file == 1) {
        config.fpga_network.num_links = 1;
    }

    // Create buffers.
    struct Buffer gpu_input_buffer[config.gpu.num_gpus];
    struct Buffer gpu_output_buffer[config.gpu.num_gpus];
    struct Buffer gpu_beamform_output_buffer[config.gpu.num_gpus];

    cl_int output_len = config.processing.num_adjusted_local_freq * config.processing.num_blocks*
                        (config.gpu.block_size*config.gpu.block_size)*2.;

    // Create the shared pool of buffer info objects; used for recording information about a
    // given frame and past between buffers as needed.
    struct InfoObjectPool pool[config.gpu.num_gpus];

    pthread_t gpu_t[config.gpu.num_gpus];
    struct gpuThreadArgs gpu_args[config.gpu.num_gpus];
    struct NullThreadArg gpu_null_thread_args[config.gpu.num_gpus];

    INFO("Starting GPU threads...");
    char buffer_name[100];
    int num_working_gpus = 1;

    for (int i = 0; i < config.gpu.num_gpus; ++i) {

        DEBUG("Creating buffers...");

        // TODO Figure out why this value currently needs to be constant.
        int links_per_gpu = 1; //num_links_per_gpu(&config, i);
        INFO("Num links for gpu[%d] = %d", i, links_per_gpu);

        create_info_pool(&pool[i], 2 * links_per_gpu * config.processing.buffer_depth,
                                    config.processing.num_adjusted_local_freq,
                                    config.processing.num_adjusted_elements);

        snprintf(buffer_name, 100, "gpu_input_buffer_%d", i);
        create_buffer(&gpu_input_buffer[i],
                      links_per_gpu * config.processing.buffer_depth,
                      config.processing.samples_per_data_set * config.processing.num_adjusted_elements *
                      config.processing.num_adjusted_local_freq * config.processing.num_data_sets,
                      1, //num_links_per_gpu(&config, i),
                      1,
                      &pool[i],
                      buffer_name);

        snprintf(buffer_name, 100, "gpu_output_buffer_%d", i);
        create_buffer(&gpu_output_buffer[i],
                      links_per_gpu * config.processing.buffer_depth,
                      output_len * config.processing.num_data_sets * sizeof(cl_int),
                      1,
                      1,
                      &pool[i],
                      buffer_name);

        snprintf(buffer_name, 100, "gpu_beamform_output_buffer_%d", i);
        /*create_buffer(&gpu_beamform_output_buffer[i],
                      links_per_gpu * config.processing.buffer_depth,
                      config.processing.samples_per_data_set * config.processing.num_data_sets *
                      config.processing.num_local_freq * 2,
                      1,
                      1,
                      &pool[i],
                      buffer_name);*/

        gpu_args[i].in_buf = &gpu_input_buffer[i];
        gpu_args[i].out_buf = &gpu_output_buffer[i];
        gpu_args[i].beamforming_out_buf = &gpu_beamform_output_buffer[i];
        gpu_args[i].gpu_id = i;
        gpu_args[i].started = 0;
        gpu_args[i].config = &config;

        if (i < num_working_gpus) {
            CHECK_ERROR( pthread_mutex_init(&gpu_args[i].lock, NULL) );
            CHECK_ERROR( pthread_cond_init(&gpu_args[i].cond, NULL) );

            DEBUG("Setting up GPU thread %d\n", i);

            pthread_create(&gpu_t[i], NULL, &gpu_thread, (void *)&gpu_args[i] );
            CHECK_ERROR( pthread_setaffinity_np(gpu_t[i], sizeof(cpu_set_t), &cpuset) );


            // Block until the OpenCL thread is read to read data.
            wait_for_gpu_thread_ready(&gpu_args[i]);

            CHECK_ERROR( pthread_mutex_destroy(&gpu_args[i].lock) );
            CHECK_ERROR( pthread_cond_destroy(&gpu_args[i].cond) );

            INFO("GPU thread %d ready.", i);
        } else {
            // Setup null threads
            gpu_null_thread_args[i].buf = &gpu_input_buffer[i];
            gpu_null_thread_args[i].config = &config;
            CHECK_ERROR( pthread_create(&gpu_t[i], NULL,
                                        &null_thread,
                                        (void *) &gpu_null_thread_args[i] ) );
            CHECK_ERROR( pthread_setaffinity_np(gpu_t[i], sizeof(cpu_set_t), &cpuset) );
        }
    }

    //sleep(5);
    INFO("Starting up network threads...");

    // Create network threads
    pthread_t network_t[config.fpga_network.num_links];
    pthread_t network_dpdk_t;
    struct networkThreadArg network_args[config.fpga_network.num_links];
    struct networkDPDKArg network_dpdk_args;

    int use_dpdk = 1;
    if (use_dpdk == 1) {

        network_dpdk_args.buf = gpu_input_buffer;
        network_dpdk_args.num_links = 4;
        network_dpdk_args.config = &config;

        CHECK_ERROR( pthread_create(&network_dpdk_t, NULL, &network_dpdk_thread,
                                    (void *)&network_dpdk_args ) );

    } else {
        for (int i = 0; i < config.fpga_network.num_links; ++i) {
            INFO("Link %d being assigned to buffer %d", i, config.fpga_network.link_map[i].gpu_id);
            network_args[i].buf = &gpu_input_buffer[config.fpga_network.link_map[i].gpu_id];
            network_args[i].ip_address = config.fpga_network.link_map[i].link_name;
            network_args[i].link_id = config.fpga_network.link_map[i].link_id;
            network_args[i].dev_id = i;
            network_args[i].num_links_in_group = num_links_in_group(&config, i);

            // Args for reading from file
            network_args[i].read_from_file = read_file;
            network_args[i].file_name = input_file_name;

            network_args[i].config = &config;

            if (no_network_test == 0) {
                CHECK_ERROR( pthread_create(&network_t[i], NULL, &network_thread,
                                            (void *)&network_args[i] ) );
            } else {
                CHECK_ERROR( pthread_create(&network_t[i], NULL, &test_network_thread,
                                            (void *)&network_args[i] ) );
            }
        }
    }

    pthread_t output_consumer_t;
    pthread_t output_network_t;

    pthread_t beamforming_comsumer_t;
    pthread_t vdif_output_t;

    struct Buffer network_output_buffer;
    struct gpuPostProcessThreadArg gpu_post_process_args;
    struct ch_acqUplinkThreadArg ch_acq_uplink_args;
    struct NullThreadArg null_thread_args;
    struct Buffer vdif_output_buffer;
    struct BeamformingPostProcessArgs beamforming_post_process_args;
    struct VDIFstreamArgs vdif_stream_args;

    if (use_ch_acq == 1) {
        int num_values = ((config.processing.num_elements *
        (config.processing.num_elements + 1)) / 2 ) *
        config.processing.num_total_freq;

        const int tcp_buffer_size = sizeof(struct tcp_frame_header) +
            num_values * sizeof(complex_int_t) +
            config.processing.num_total_freq * sizeof(struct per_frequency_data) +
            config.processing.num_total_freq * config.processing.num_elements * sizeof(struct per_element_data) +
            num_values * sizeof(uint8_t);

        // TODO config file this.
        const int network_buffer_depth = 4;

        create_buffer(&network_output_buffer, network_buffer_depth, tcp_buffer_size,
                      1, 1, pool, "network_output_buffer");

        // The thread which creates output frame.
        gpu_post_process_args.in_buf = gpu_output_buffer;
        gpu_post_process_args.out_buf = &network_output_buffer;
        gpu_post_process_args.config = &config;
        CHECK_ERROR( pthread_create(&output_consumer_t, NULL,
                                    &gpu_post_process_thread,
                                    (void *) &gpu_post_process_args ) );
        CHECK_ERROR( pthread_setaffinity_np(output_consumer_t, sizeof(cpu_set_t), &cpuset) );

        if (config.ch_master_network.disable_upload == 0) {
            // The thread which sends it with TCP to ch_acq.
            ch_acq_uplink_args.buf = &network_output_buffer;
            ch_acq_uplink_args.config = &config;
            CHECK_ERROR( pthread_create(&output_network_t, NULL,
                                        &ch_acq_uplink_thread,
                                        (void *) &ch_acq_uplink_args ) );
        } else {
            // Drop the data.
            null_thread_args.buf = &network_output_buffer;
            null_thread_args.config = &config;
            CHECK_ERROR( pthread_create(&output_network_t, NULL,
                                        &null_thread,
                                        (void *) &null_thread_args ) );
        }
        //CHECK_ERROR( pthread_setaffinity_np(output_network_t, sizeof(cpu_set_t), &cpuset) );

        // The beamforming thread
        if (config.gpu.use_beamforming == 1) {
            create_buffer(&vdif_output_buffer, network_buffer_depth, 625*16*5032,
                          1, 1, pool, "vdif_output_buffer");

            // The thread which creates output frame.
            beamforming_post_process_args.in_buf = gpu_beamform_output_buffer;
            beamforming_post_process_args.out_buf = &vdif_output_buffer;
            beamforming_post_process_args.config = &config;
            CHECK_ERROR( pthread_create(&beamforming_comsumer_t, NULL,
                                        &beamforming_post_process,
                                        (void *) &beamforming_post_process_args ) );

            // The thread which sends it with TCP to ch_acq.
            vdif_stream_args.buf = &vdif_output_buffer;
            vdif_stream_args.config = &config;
            CHECK_ERROR( pthread_create(&vdif_output_t, NULL,
                                        &vdif_stream,
                                        (void *) &vdif_stream_args ) );
        }
    } else  {
        // TODO add local file output in some form here.
    }

    // Join with threads.
    int * ret;

    for (int i = 0; i < config.gpu.num_gpus; ++i) {
        CHECK_ERROR( pthread_join(gpu_t[i], (void **) &ret) );
    }
    if (use_dpdk == 1) {
        CHECK_ERROR( pthread_join(network_dpdk_t, (void **) &ret) );
    } else {
        for (int i = 0; i < config.fpga_network.num_links; ++i) {
            CHECK_ERROR( pthread_join(network_t[i], (void **) &ret) );
        }
    }

    CHECK_ERROR( pthread_join(output_consumer_t, (void **) &ret) );

    if (use_ch_acq == 1) {
        CHECK_ERROR( pthread_join(output_network_t, (void **) &ret) );
    }

    if (config.gpu.use_beamforming == 1) {
        CHECK_ERROR( pthread_join(beamforming_comsumer_t, (void **) &ret) );
        CHECK_ERROR( pthread_join(vdif_output_t, (void **) &ret) );
    }

    INFO("kotekan shutdown successfully.");

    closelog();

    return 0;
}
