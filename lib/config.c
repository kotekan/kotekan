
#include <jansson.h>
#include <string.h>
#include <assert.h>

#include "config.h"
#include "errors.h"

int parse_processing_config(struct Config* config, json_t * json)
{
    int error = 0;
    json_t * product_remap;

    error = json_unpack(json, "{s:i, s:i, s:i, s:i, s:i, s:i, s:i, s:o, s:i}",
        "num_elements", &config->processing.num_elements,
        "num_local_freq", &config->processing.num_local_freq,
        "num_total_freq", &config->processing.num_total_freq,
        "samples_per_data_set", &config->processing.samples_per_data_set,
        "num_data_sets", &config->processing.num_data_sets,
        "buffer_depth", &config->processing.buffer_depth,
        "num_gpu_frames", &config->processing.num_gpu_frames,
        "product_remap", &product_remap,
        "data_limit", &config->processing.data_limit);

    if (error) {
        ERROR("Error parsing processing config.");
        return error;
    }

    int remap_size = json_array_size(product_remap);

    if (remap_size != config->processing.num_elements) {
        ERROR("The remap array must have the same size as the number of elements. array size %d, num_elements %d",
            remap_size, config->processing.num_elements);
        return -2;
    }

    config->processing.product_remap = malloc(remap_size * sizeof(int));
    assert(config->processing.product_remap != NULL);

    for(int i = 0; i < remap_size; ++i) {
        config->processing.product_remap[i] = json_integer_value(json_array_get(product_remap, i));
    }

    config->processing.inverse_product_remap = malloc(remap_size * sizeof(int));
    CHECK_MEM(config->processing.inverse_product_remap);
    // Given a channel ID, where is it in FPGA order.
    for(int i = 0; i < remap_size; ++i) {
        config->processing.inverse_product_remap[config->processing.product_remap[i]] = i;
    }

    // Special case for 16-element version
    if (config->processing.num_elements < 32) {
        config->processing.num_adjusted_elements = 32;
        config->processing.num_adjusted_local_freq = 64;
    } else {
        config->processing.num_adjusted_elements = config->processing.num_elements;
        config->processing.num_adjusted_local_freq = config->processing.num_local_freq;
    }

    // TODO Hard coded constants are BAD!
    config->processing.num_blocks = (config->processing.num_adjusted_elements / 32) *
        (config->processing.num_adjusted_elements / 32 + 1) / 2.;

    return 0;
}

int parse_ch_master_networking_config(struct Config* config, json_t * json)
{
    int error = 0;
    char * collection_server_ip;

    error = json_unpack(json, "{s:s, s:i, s:i}",
        "collection_server_ip", &collection_server_ip,
        "collection_server_port", &config->ch_master_network.collection_server_port,
        "disable_upload", &config->ch_master_network.disable_upload);

    if (error) {
        ERROR("Error parsing ch_master_network config, check config file, error: %d", error);
        return error;
    }

    if (collection_server_ip == NULL) {
        ERROR("The collection server IP address in the config file is not a valid string.");
        return -1;
    }

    config->ch_master_network.collection_server_ip = strdup(collection_server_ip);

    return 0;
}

int parse_gpu_config(struct Config* config, json_t * json)
{
    int error = 0;
    json_t * kernels;

    error = json_unpack(json, "{s:o, s:i, s:i, s:i, s:i, s:i, s:i, s:i}",
        "kernels", &kernels,
        "use_timeshift", &config->gpu.use_time_shift,
        "ts_element_offset", &config->gpu.ts_element_offset,
        "ts_num_elem_to_shift", &config->gpu.ts_num_elem_to_shift,
        "ts_samples_to_shift", &config->gpu.ts_samples_to_shift,
        "num_gpus", &config->gpu.num_gpus,
        "block_size", &config->gpu.block_size,
        "use_beamforming", &config->gpu.use_beamforming);

    if (error) {
        ERROR("Error parsing gpu config");
        return error;
    }

    config->gpu.num_kernels = json_array_size(kernels);
    if (config->gpu.num_kernels <= 0) {
        ERROR("No kernel file names given");
        return -1;
    }
    config->gpu.kernels = malloc(config->gpu.num_kernels * sizeof(char *));
    assert(config->gpu.kernels != NULL);

    for (int i = 0; i < config->gpu.num_kernels; ++i) {
        const char * kernel_file_name = json_string_value(json_array_get(kernels, i));

        if (kernel_file_name == NULL) {
            ERROR("A kernel file name was invalid, or could not be parsed, check config file");
            return -1;
        }
        // Make a copy of the string for the config object.
        config->gpu.kernels[i] = strdup(kernel_file_name);
    }

    return 0;
}

int parse_fpga_network_config(struct Config* config, json_t * json)
{
    int error = 0;
    json_t * link_map;

    error = json_unpack(json, "{s:i, s:i, s:i, s:i, s:i, s:o}",
        "num_links", &config->fpga_network.num_links,
        "port_number", &config->fpga_network.port_number,
        "timesamples_per_packet", &config->fpga_network.timesamples_per_packet,
        "udp_frame_size", &config->fpga_network.udp_frame_size,
        "udp_packet_size", &config->fpga_network.udp_packet_size,
        "link_map", &link_map);

    if (error) {
        ERROR("Error parsing fpga_network config section.");
        return -1;
    }

    int num_links_in_map = json_array_size(link_map);
    if (num_links_in_map != config->fpga_network.num_links) {
        ERROR("The size of the link map must be equal to the number of links");
        return -2;
    }

    config->fpga_network.link_map = malloc(num_links_in_map * sizeof(struct LinkMap));
    assert(config->fpga_network.link_map != NULL);

    for (int i = 0; i < num_links_in_map; ++i) {
        json_t * link;
        char * link_name;

        link = json_array_get(link_map, i);
        if (link == NULL) {
            ERROR("Error reading link_map check config file");
            return -2;
        }

        error = json_unpack(link, "{s:s, s:i, s:i}",
            "link_name", &link_name,
            "gpu_id", &config->fpga_network.link_map[i].gpu_id,
            "link_id", &config->fpga_network.link_map[i].link_id);

        if (link_name == NULL) {
            ERROR("Link name is null for link %d in config file.", i);
            return -2;
        }

        config->fpga_network.link_map[i].link_name = strdup(link_name);
        config->fpga_network.link_map[i].stream_id = 0;
    }

    return 0;
}

int parse_beamforming_config(struct Config* config, json_t * json) {
    int error = 0;
    char * server_ip;
    json_t * element_mask, * element_positions;

    error = json_unpack(json, "{s:s, s:i, s:F, s:F, s:F, s:F, s:o, s:o, s:i}",
                        "vdif_server_ip", &server_ip,
                        "vdif_port", &config->beamforming.vdif_port,
                        "ra", &config->beamforming.ra,
                        "dec", &config->beamforming.dec,
                        "instrument_lat", &config->beamforming.instrument_lat,
                        "instrument_long", &config->beamforming.instrument_long,
                        "element_mask", &element_mask,
                        "element_positions", &element_positions,
                        "bit_shift_factor", &config->beamforming.bit_shift_factor);

    if (error) {
        ERROR("Error parsing beamforming config, check config file, error: %d", error);
        return error;
    }

    if (server_ip == NULL) {
        ERROR("The vdif_server_ip address in the config file is not a valid string.");
        return -1;
    }
    config->beamforming.vdif_server_ip = strdup(server_ip);

    config->beamforming.num_masked_elements = json_array_size(element_mask);
    if (config->beamforming.num_masked_elements > 0) {
        config->beamforming.element_mask = malloc(config->beamforming.num_masked_elements * sizeof(int));
        CHECK_MEM(config->beamforming.element_mask);

        for (int i = 0; i < config->beamforming.num_masked_elements; ++i) {
            config->beamforming.element_mask[i] = json_integer_value(json_array_get(element_mask, i));
        }
    }

    int num_positions = json_array_size(element_positions);
    if (config->processing.num_elements * 2 != num_positions) {
        ERROR("The number of element positions must match the number of elements, num_positions %d", num_positions);
        return -1;
    }
    config->beamforming.element_positions = malloc(num_positions * sizeof(float));
    CHECK_MEM(config->beamforming.element_positions);

    for (int i = 0; i < num_positions / 2; ++i) {
        config->beamforming.element_positions[i*2] =
            json_number_value(json_array_get(element_positions, 2*config->processing.product_remap[i]));
        config->beamforming.element_positions[i*2 + 1] =
            json_number_value(json_array_get(element_positions, 2*config->processing.product_remap[i] + 1));
    }

    return 0;
}

int parse_config(struct Config* config, json_t * json)
{
    int error = 0;

    char * type;
    json_unpack(json, "{s:s}", "type", &type);

    if (strcmp(type, "config") != 0) {
        ERROR("The json object isn't a config object");
        return -1;
    }

    json_t * gpu_json, * fpga_network_json, * processing_json,
            * ch_master_network_json, * beamforming_json;

    error = json_unpack(json, "{s:o, s:o, s:o, s:o, s:o}",
                        "gpu", &gpu_json,
                        "fpga_network", &fpga_network_json,
                        "processing", &processing_json,
                        "ch_master_network", &ch_master_network_json,
                        "beamforming", &beamforming_json);

    if (error) {
        ERROR("Error processing config root structure");
        return error;
    }

    error |= parse_processing_config(config, processing_json);
    error |= parse_ch_master_networking_config(config, ch_master_network_json);
    error |= parse_gpu_config(config, gpu_json);
    error |= parse_fpga_network_config(config, fpga_network_json);
    error |= parse_beamforming_config(config, beamforming_json);

    return error;
}

void delete_config(struct Config* config)
{

}

int load_config_from_file(struct Config* config, char* file_name)
{
    json_t *json;
    json_error_t json_error;
    int error;

    json = json_load_file(file_name, 0, &json_error);
    if (!json) {
        ERROR("The config file could not be read, or is invalid json.");
        return -1;
    }

    error = parse_config(config, json);

    return error;
}

void print_config(struct Config* config)
{
    // CH_MASTER Network Section
    INFO("config.ch_master_network.collection_server_ip = %s",
         config->ch_master_network.collection_server_ip);
    INFO("config.ch_master_network.collection_server_port = %d",
         config->ch_master_network.collection_server_port);

    // GPU section
    INFO("config.gpu.num_gpus = %d", config->gpu.num_gpus);
    INFO("config.gpu.block_size = %d", config->gpu.block_size);
    INFO("config.gpu.num_kernels = %d", config->gpu.num_kernels);
    for (int i = 0; i < config->gpu.num_kernels; ++i) {
        INFO("config.gpu.kernels[%d] = %s", i, config->gpu.kernels[i]);
    }
    INFO("config.gpu.use_time_shift = %d", config->gpu.use_time_shift);
    INFO("config.gpu.ts_element_offset = %d", config->gpu.ts_element_offset);
    INFO("config.gpu.ts_num_elem_to_shift = %d", config->gpu.ts_num_elem_to_shift);
    INFO("config.gpu.ts_samples_to_shift = %d", config->gpu.ts_samples_to_shift);

    // Processing section
    INFO("config.processing.num_elements = %d", config->processing.num_elements);
    INFO("config.processing.num_local_freq = %d", config->processing.num_local_freq);
    INFO("config.processing.num_total_freq = %d", config->processing.num_total_freq);
    INFO("config.processing.samples_per_data_set = %d", config->processing.samples_per_data_set);
    INFO("config.processing.num_data_sets = %d", config->processing.num_data_sets);
    INFO("config.processing.buffer_depth = %d", config->processing.buffer_depth);
    INFO("config.processing.num_adjusted_elements = %d", config->processing.num_adjusted_elements);
    INFO("config.processing.num_adjusted_local_freq = %d", config->processing.num_adjusted_local_freq);
    INFO("config.processing.num_blocks = %d", config->processing.num_blocks);

    for (int i = 0; i < config->processing.num_elements; ++i) {
        //INFO("config.processing.product_remap[%d] = %d", i, config->processing.product_remap[i]);
    }

    // FPGA Network Section
    INFO("config.fpga_network.num_links = %d", config->fpga_network.num_links);
    INFO("config.fpga_network.port_number = %d", config->fpga_network.port_number);
    INFO("config.fpga_network.timesamples_per_packet = %d", config->fpga_network.timesamples_per_packet);
    INFO("config.fpga_network.udp_packet_size = %d", config->fpga_network.udp_packet_size);
    INFO("config.fpga_network.udp_frame_size = %d", config->fpga_network.udp_frame_size);

    for (int i = 0; i < config->fpga_network.num_links; ++i) {
        INFO("config.fpga_network.link_map[%d].link_name = %s", i,
             config->fpga_network.link_map[i].link_name);
        INFO("config.fpga_network.link_map[%d].gpu_id = %d", i,
             config->fpga_network.link_map[i].gpu_id);
        INFO("config.fpga_network.link_map[%d].stream_id = %d", i,
             config->fpga_network.link_map[i].stream_id);
        INFO("config.fpga_network.link_map[%d].link_id = %d", i,
             config->fpga_network.link_map[i].link_id);
    }

    // Beamforming section
    INFO("config.beamforming.num_masked_elements = %d", config->beamforming.num_masked_elements);
    INFO("config.beamforming.bit_shift_factor = %d", config->beamforming.bit_shift_factor);
    INFO("config.beamforming.ra = %f", config->beamforming.ra);
    INFO("config.beamforming.dec = %f", config->beamforming.dec);
    INFO("config.beamforming.instrument_lat = %f", config->beamforming.instrument_lat);
    INFO("config.beamforming.instrument_long = %f", config->beamforming.instrument_long);

}

int num_links_in_group(struct Config* config, const unsigned int link_id)
{
    assert(link_id < config->fpga_network.num_links);

    int gpu_id = config->fpga_network.link_map[link_id].gpu_id;
    int num_links = 0;

    for (int i = 0; i < config->fpga_network.num_links; ++i) {
        if (config->fpga_network.link_map[i].gpu_id == gpu_id)
            num_links++;
    }

    return num_links;
}

int num_links_per_gpu(struct Config* config, const unsigned int gpu_id)
{
    assert(gpu_id < config->gpu.num_gpus);

    int num_links = 0;

    for (int i = 0; i < config->fpga_network.num_links; ++i) {
        if (config->fpga_network.link_map[i].gpu_id == gpu_id)
            num_links++;
    }

    return num_links;
}
