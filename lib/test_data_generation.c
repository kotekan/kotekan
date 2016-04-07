
#include "test_data_generation.h"

#include <stdlib.h>

int offset_and_clip_value(int input_value, int offset_value, int min_val, int max_val){
    int offset_and_clipped = input_value + offset_value;
    if (offset_and_clipped > max_val)
        offset_and_clipped = max_val;
    else if (offset_and_clipped < min_val)
        offset_and_clipped = min_val;
    return(offset_and_clipped);
}


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
                            unsigned char *packed_data_set){

    //sfmt_t sfmt; //for the Mersenne Twister
    if (single_frequency > num_frequencies || single_frequency < 0)
        single_frequency = ALL_FREQUENCIES;

    //printf("single_frequency: %d \n",single_frequency);
    default_real =offset_and_clip_value(default_real,8,0,15);
    default_imaginary = offset_and_clip_value(default_imaginary,8,0,15);
    initial_real = offset_and_clip_value(initial_real,8,0,15);
    initial_imaginary = offset_and_clip_value(initial_imaginary,8,0,15);
    unsigned char clipped_offset_default_real = (unsigned char) default_real;
    unsigned char clipped_offset_default_imaginary = (unsigned char) default_imaginary;
    unsigned char clipped_offset_initial_real = (unsigned char) initial_real;
    unsigned char clipped_offset_initial_imaginary = (unsigned char) initial_imaginary;

    //printf("clipped_offset_initial_real: %d, clipped_offset_initial_imaginary: %d, clipped_offset_default_real: %d, clipped_offset_default_imaginary: %d\n", clipped_offset_initial_real, clipped_offset_initial_imaginary, clipped_offset_default_real, clipped_offset_default_imaginary);

    if (generation_Type == GENERATE_DATASET_RANDOM_SEEDED){
        //sfmt_init_gen_rand(&sfmt, random_seed);
        srand(random_seed);
    }

    for (int k = 0; k < num_timesteps; k++){
        //printf("k: %d\n",k);
        if (generation_Type == GENERATE_DATASET_RANDOM_SEEDED && GEN_REPEAT_RANDOM){
            //sfmt_init_gen_rand(&sfmt, random_seed);
            srand(random_seed);
        }
        for (int j = 0; j < num_frequencies; j++){
            //printf("j: %d\n",j);
            for (int i = 0; i < num_elements; i++){
                int currentAddress = k*num_frequencies*num_elements + j*num_elements + i;
                unsigned char new_real;
                unsigned char new_imaginary;
                switch (generation_Type){
                    case GENERATE_DATASET_CONSTANT:
                        new_real = 8; //clipped_offset_initial_real;
                        new_imaginary = 8; //clipped_offset_initial_imaginary;
                        break;
                    case GENERATE_DATASET_RAMP_UP:
                        new_real = (j+clipped_offset_initial_real+i)%16;
                        new_imaginary = (j+clipped_offset_initial_imaginary+i)%16;
                        break;
                    case GENERATE_DATASET_RAMP_DOWN:
                        new_real = 15-((j+clipped_offset_initial_real+i)%16);
                        new_imaginary = 15 - ((j+clipped_offset_initial_imaginary+i)%16);
                        break;
                    case GENERATE_DATASET_RANDOM_SEEDED:
                        new_real = (unsigned char)rand()%16; //to put the pseudorandom value in the range 0-15
                        new_imaginary = (unsigned char)rand()%16;
                        break;
                    default: //shouldn't happen, but in case it does, just assign the default values everywhere
                        new_real = clipped_offset_default_real;
                        new_imaginary = clipped_offset_default_imaginary;
                        break;
                }

                if (single_frequency == ALL_FREQUENCIES){
                    packed_data_set[currentAddress] = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                }
                else{
                    if (j == single_frequency)
                        packed_data_set[currentAddress] = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                    else
                        packed_data_set[currentAddress] = ((clipped_offset_default_real<<4) & 0xF0) + (clipped_offset_default_imaginary & 0x0F);
                }
                //printf("%d ",data_set[currentAddress]);
            }
        }
    }

    return;
}
