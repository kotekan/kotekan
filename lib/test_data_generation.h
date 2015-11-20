
#ifndef TEST_DATA_GENERATION
#define TEST_DATA_GENERATION

//enumerations/definitions: don't change
#define GENERATE_DATASET_CONSTANT       1u
#define GENERATE_DATASET_RAMP_UP        2u
#define GENERATE_DATASET_RAMP_DOWN      3u
#define GENERATE_DATASET_RANDOM_SEEDED  4u
#define ALL_FREQUENCIES                -1

//parameters for data generator: you can change these. (Values will be shifted and clipped as needed, so these are signed 4bit numbers for input)
#define GEN_TYPE                        GENERATE_DATASET_CONSTANT
#define GEN_DEFAULT_SEED                42u
#define GEN_DEFAULT_RE                  0u
#define GEN_DEFAULT_IM                  0u
#define GEN_INITIAL_RE                  0
#define GEN_INITIAL_IM                  0
#define GEN_FREQ                        ALL_FREQUENCIES
#define GEN_REPEAT_RANDOM               1u

#define CHECKING_VERBOSE                0u

#ifdef __cplusplus
extern "C" {
#endif

int offset_and_clip_value(int input_value, int offset_value, int min_val, int max_val);


void generate_char_data_set(int generation_Type,
                            int random_seed,
                            int default_real,
                            int default_imaginary,
                            int initial_real,
                            int initial_imaginary,
                            int single_frequency,
                            int num_timesteps,
                            int num_frequencies,
                            int num_elements,
                            unsigned char *packed_data_set);

#ifdef __cplusplus
}
#endif

#endif