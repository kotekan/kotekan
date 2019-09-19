//  gcc -std=gnu99 -O3 -Wall fourierTesting_128_upchannelization.c -I$AMDAPPSDKROOT/include -lOpenCL -lm -L$AMDAPPSDKROOT/lib/x86_64/ -o fourierTest128upchan

#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include <getopt.h>
#include <sys/time.h>
#define OPENCL_FILENAME_1   "upchannelize_flip.cl"
#define NUM_CL_FILES        1
#define N_QUEUES            1
#define FLT_EPSILON 0.00015
//enumerations/definitions: don't change
#define GENERATE_DATASET_CONSTANT       1u
#define GENERATE_DATASET_RAMP_UP        2u
#define GENERATE_DATASET_RAMP_DOWN      3u
#define GENERATE_DATASET_RANDOM_SEEDED  4u
#define ALL_FREQUENCIES                -1

//parameters for data generator: you can change these. (Values will be shifted and clipped as needed, so these are signed 4bit numbers for input)
#define GEN_TYPE                        GENERATE_DATASET_RANDOM_SEEDED
#define GEN_DEFAULT_SEED                42u
#define GEN_DEFAULT_RE                  0u
#define GEN_DEFAULT_IM                  0u
#define GEN_INITIAL_RE                  0u //-8
#define GEN_INITIAL_IM                  0u //7
#define GEN_FREQ                        ALL_FREQUENCIES
#define GEN_REPEAT_RANDOM               0u

#define XOR_MANUAL                      0u
#define DEBUG_GENERATOR                 0u

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

#define HI_NIBBLE(b)                    (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b)                    ((b) & 0x0F)

//from Numerical Recipes in C
/******************************************************************************/
void four1(float *data, unsigned long nn, int isign)
/*******************************************************************************
Replaces data[0..2*nn-1] by its discrete Fourier transform, if isign is input as
1; or replaces data[0..2*nn-1] by nn times its inverse discrete Fourier transform,
if isign is input as -1.  data is a complex array of length nn or, equivalently,
a real array of length 2*nn.  nn MUST be an integer power of 2
*******************************************************************************/
{
    unsigned long n,mmax,m,j,istep,i;
    float wtemp,wr,wpr,wpi,wi,theta;
    float tempr,tempi;

    if (nn<2 || nn&(nn-1)){
        perror("Length must be a power of 2");
        exit(-4);
    }
    n=nn << 1;
    j=1;
    for (i=1;i<n;i+=2) { /* This is the bit-reversal section of the routine. */
        if (j > i) {
            SWAP(data[j-1],data[i-1]); /* Exchange the two complex numbers. */
            SWAP(data[j],data[i]);
        }
        m=nn;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    mmax=2;
    //for (int k= 0; k < 1; k++){
    while (n > mmax) { /* Outer loop executed log2 nn times. */
        istep=mmax << 1;
        theta=isign*(6.28318530717959/mmax); /* Initialize the trigonometric recurrence. */
        wtemp=sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi=sin(theta);
        wr=1.0;
        wi=0.0;
        //printf("wpr %8f, wpi %8f, wr %8f, wi %8f\n", wpr, wpi, wr, wi);
        for (m=1;m<mmax;m+=2) { /* Here are the two nested inner loops. */
            //printf("m: %02ld mmax: %02ld, wr: %8f, wi: %8f, theta:%8f\n",m, mmax, wr, wi, theta/6.28318530717959);
            for (i=m;i<=n;i+=istep) {
                j=i+mmax; /* This is the Danielson-Lanczos formula. */
                //printf("i: %02ld, j: %02ld, mmax: %02ld, index 1: %02ld index 2: %02ld\n", i, j, mmax, (i-1)/2, (j-1)/2);
                //printf("     data[%ld]_Re = data[%ld]_Re + (%f * data[%ld]_Re - %f * data[%ld]_Im)\n",(i-1)/2, (i-1)/2, wr, (j-1)/2, wi, (j-1)/2);
                //printf("     data[%ld]_Im = data[%ld]_Im + (%f * data[%ld]_Im + %f * data[%ld]_Re)\n",(i-1)/2, (i-1)/2, wr, (j-1)/2, wi, (j-1)/2);
                //printf("     data[%ld]_Re = data[%ld]_Re - (%f * data[%ld]_Re - %f * data[%ld]_Im)\n",(j-1)/2, (i-1)/2, wr, (j-1)/2, wi, (j-1)/2);
                //printf("     data[%ld]_Im = data[%ld]_Im - (%f * data[%ld]_Im + %f * data[%ld]_Re)\n\n",(j-1)/2, (i-1)/2, wr, (j-1)/2, wi, (j-1)/2);

                tempr=wr*data[j-1]-wi*data[j];
                tempi=wr*data[j]+wi*data[j-1];
                //printf("%f %f %f %f\n",data[i-1], data[i], data[j-1], data[j]);
                data[j-1]=data[i-1]-tempr;
                data[j]=data[i]-tempi;
                data[i-1] += tempr;
                data[i] += tempi;
                //printf("%f %f %f %f\n",data[i-1], data[i], data[j-1], data[j]);
            }
            wr=(wtemp=wr)*wpr-wi*wpi+wr; /* Trigonometric recurrence. */
            wi=wi*wpr+wtemp*wpi+wi;
        }
        mmax=istep;
    }
}

//from Numerical Recipes in C
/******************************************************************************/
void four1double(double *data, unsigned long nn, int isign)
/*******************************************************************************
Replaces data[0..2*nn-1] by its discrete Fourier transform, if isign is input as
1; or replaces data[0..2*nn-1] by nn times its inverse discrete Fourier transform,
if isign is input as -1.  data is a complex array of length nn or, equivalently,
a real array of length 2*nn.  nn MUST be an integer power of 2
*******************************************************************************/
{
    unsigned long n,mmax,m,j,istep,i;
    double wtemp,wr,wpr,wpi,wi,theta;
    double tempr,tempi;

    if (nn<2 || nn&(nn-1)){
        perror("Length must be a power of 2");
        exit(-4);
    }
    n=nn << 1;
    j=1;
    for (i=1;i<n;i+=2) { /* This is the bit-reversal section of the routine. */
        if (j > i) {
            SWAP(data[j-1],data[i-1]); /* Exchange the two complex numbers. */
            SWAP(data[j],data[i]);
        }
        m=nn;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    mmax=2;
    //for (int k= 0; k < 1; k++){
    while (n > mmax) { /* Outer loop executed log2 nn times. */
        istep=mmax << 1;
        theta=isign*(6.28318530717959/mmax); /* Initialize the trigonometric recurrence. */
        wtemp=sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi=sin(theta);
        wr=1.0;
        wi=0.0;
        //printf("wpr %8f, wpi %8f, wr %8f, wi %8f\n", wpr, wpi, wr, wi);
        for (m=1;m<mmax;m+=2) { /* Here are the two nested inner loops. */
            //printf("m: %02ld mmax: %02ld, wr: %8f, wi: %8f, theta:%8f\n",m, mmax, wr, wi, theta/6.28318530717959);
            for (i=m;i<=n;i+=istep) {
                j=i+mmax; /* This is the Danielson-Lanczos formula. */
                //printf("i: %02ld, j: %02ld, mmax: %02ld, index 1: %02ld index 2: %02ld\n", i, j, mmax, (i-1)/2, (j-1)/2);
                //printf("     data[%ld]_Re = data[%ld]_Re + (%f * data[%ld]_Re - %f * data[%ld]_Im)\n",(i-1)/2, (i-1)/2, wr, (j-1)/2, wi, (j-1)/2);
                //printf("     data[%ld]_Im = data[%ld]_Im + (%f * data[%ld]_Im + %f * data[%ld]_Re)\n",(i-1)/2, (i-1)/2, wr, (j-1)/2, wi, (j-1)/2);
                //printf("     data[%ld]_Re = data[%ld]_Re - (%f * data[%ld]_Re - %f * data[%ld]_Im)\n",(j-1)/2, (i-1)/2, wr, (j-1)/2, wi, (j-1)/2);
                //printf("     data[%ld]_Im = data[%ld]_Im - (%f * data[%ld]_Im + %f * data[%ld]_Re)\n\n",(j-1)/2, (i-1)/2, wr, (j-1)/2, wi, (j-1)/2);

                tempr=wr*data[j-1]-wi*data[j];
                tempi=wr*data[j]+wi*data[j-1];
                //printf("%f %f %f %f\n",data[i-1], data[i], data[j-1], data[j]);
                data[j-1]=data[i-1]-tempr;
                data[j]=data[i]-tempi;
                data[i-1] += tempr;
                data[i] += tempi;
                //printf("%f %f %f %f\n",data[i-1], data[i], data[j-1], data[j]);
            }
            wr=(wtemp=wr)*wpr-wi*wpi+wr; /* Trigonometric recurrence. */
            wi=wi*wpr+wtemp*wpi+wi;
        }
        mmax=istep;
    }
}

int bit_reverse_binary_rep_out(int in_val, int bits_to_reverse){
    int output_number = 0;
    for (int i = 0; i < bits_to_reverse; i++){
        output_number = output_number*10 + ((in_val>>i)&0x1);
    }
    return output_number;


}

int bit_reverse_binary(int in_val, int bits_to_reverse){
    int output_number = 0;
    for (int i = 0; i < bits_to_reverse; i++){
        output_number = output_number*2 +( (in_val>>i)&0x1);
    }
    return output_number;
}

void fft_stepping(double *data, unsigned long transform_length, int isign, int stop_level)
/*******************************************************************************
Replaces data[0..2*transform_length-1] by its discrete Fourier transform, if isign is input as
1; or replaces data[0..2*transform_length-1] by transform_length times its inverse discrete Fourier transform,
if isign is input as -1.  data is a complex array of length transform_length or, equivalently,
a real array of length 2*transform_length.  transform_length MUST be an integer power of 2
*******************************************************************************/
//this routine should be more or less functionally the same as Numerical Recipes' version,
//but allows a user to stop at different stages in the transform easily (for debugging GPU implementations)
{
    unsigned long n,m,j,i;
    double wr,wi,theta;
    double tempr,tempi;

    //check that the transform_length is a power of 2 using a bit manipulation trick
    if (transform_length<2 || transform_length&(transform_length-1)){
        perror("Length must be a power of 2");
        exit(-4);
    }
    n=transform_length << 1;
    j=1;
    for (i=1;i<n;i+=2) { /* This is the bit-reversal section of the routine. */
        if (j > i) {
            SWAP(data[j-1],data[i-1]); /* Exchange the two complex numbers. */
            SWAP(data[j],data[i]);
        }
        m=transform_length;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    long int step_stop; //question: why did the routine work when the step_stop was >= transform_length
    if (stop_level < 0) //neg values mean do the whole sequence; the last stage has pairs half the transform length apart
        step_stop = transform_length/2;
    else
        step_stop =pow(2,stop_level);

    if (step_stop > transform_length/2) //safety net for transform lengths re my above puzzlement
        step_stop = transform_length/2;

    for (long int step_size = 1; step_size <= step_stop ; step_size +=step_size) {
        theta = isign*3.141592654/(step_size); //I think it should actually be isign * 2 * pi / (2 * step_size), but the 2 on top and bottom cancel out

        for (int index = 0; index < transform_length; index += step_size*2){ //the trig pattern repeats every step_size*2 entries
            for (int minor_index = 0; minor_index < step_size; minor_index++){ //this inner loop takes cares of the pairs in a set of entries step_size*2 entries
                wr = cos(minor_index*theta);
                wi = sin(minor_index*theta);
                int first_index = (index+minor_index)*2;// *2 for the Re,Im pairs
                int second_index = first_index + step_size*2; //again *2 to account for the pairs
                tempr = wr*data[second_index]-wi*data[second_index+1];
                tempi = wi*data[second_index]+wr*data[second_index+1];
                data[second_index    ]  = data[first_index  ]-tempr;
                data[second_index + 1]  = data[first_index+1]-tempi;
                data[first_index     ] += tempr;
                data[first_index  + 1] += tempi;
            }
        }
    }


}


///Discrete Fourier Transform
void discreteFourierTransform(double * data_array, int nn, int sign){
    double *temp_array;
    double pi = atan(1.)*4.;
    //printf("\nPI: %f\n",pi);

    temp_array = (double *) malloc(nn*2*sizeof(double));
    if (temp_array == NULL){
        printf("Error allocating memory");
        exit(-1);
    }
    for (int i=0; i < nn*2; i++)
        temp_array[i] = 0.;

    for (int n = 0; n < nn; n++){
        for (int k = 0; k < nn; k++){
            double theta = 2. * pi * k * n / nn * sign;
            double temp_real = cos(theta);
            double temp_imag = sin(theta);
            temp_array[n*2]   += data_array[k*2]*temp_real - data_array[k*2+1]*temp_imag;
            temp_array[n*2+1] += data_array[k*2]*temp_imag + data_array[k*2+1]*temp_real;

        }
    }

    //copy results
    for (int i = 0; i < nn*2; i++){
        data_array[i] = temp_array[i];
    }

    free (temp_array);
}


double e_time(void){
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

void fourier16complex(float *data, int sign){

}


//number comparison by Bruce Dawson: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
//bools changed to ints
int AlmostEqualRelativeAndAbs(float A, float B,
                                float maxDiff, float maxRelDiff){
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    float diff = fabs(A - B);
    if (diff <= maxDiff)
        return 1;

    A = fabs(A);
    B = fabs(B);
    float largest = (B > A) ? B : A;

    if (diff <= largest * maxRelDiff)
        return 1;
    return 0;
}

double RelativeAndAbsdouble(double A, double B,
                                double maxDiff){
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    if (((float)B)==(float)(0.) || ((float)B)== (float)(-0.))
        return A;
    double diff = fabs(A - B);
    if (diff <= maxDiff)
        return 0;


    A = fabs(A);
    B = fabs(B);
    double largest = (B > A) ? B : A;

    if (largest <= maxDiff)
        return largest/maxDiff;


    return (A -B )/B;

}


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
                            int num_data_sets,
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
    unsigned char temp_output;


    //printf("clipped_offset_initial_real: %d, clipped_offset_initial_imaginary: %d, clipped_offset_default_real: %d, clipped_offset_default_imaginary: %d\n", clipped_offset_initial_real, clipped_offset_initial_imaginary, clipped_offset_default_real, clipped_offset_default_imaginary);
    for (int m = 0; m < num_data_sets; m++){
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
                if (DEBUG_GENERATOR && k == 0)
                    printf("j: %d Vals: ",j);
                for (int i = 0; i < num_elements; i++){
                    int currentAddress = m*num_timesteps*num_frequencies*num_elements +k*num_frequencies*num_elements + j*num_elements + i;
                    unsigned char new_real;
                    unsigned char new_imaginary;
                    switch (generation_Type){
                        case GENERATE_DATASET_CONSTANT:
                            new_real = clipped_offset_initial_real;
                            new_imaginary = clipped_offset_initial_imaginary;
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
                            new_real = rand()%16; //to put the pseudorandom value in the range 0-15
                            new_imaginary = rand()%16;
                            break;
                        default: //shouldn't happen, but in case it does, just assign the default values everywhere
                            new_real = clipped_offset_default_real;
                            new_imaginary = clipped_offset_default_imaginary;
                            break;
                    }

                    if (single_frequency == ALL_FREQUENCIES){
                        temp_output = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                        if (XOR_MANUAL){
                            temp_output = temp_output ^ 0x88; //bit flip on sign bit shifts the value by 8: makes unsigned into signed and vice versa.  Currently turns back into signed
                        }
                        packed_data_set[currentAddress] = temp_output;
                    }
                    else{
                        if (j == single_frequency){
                            temp_output = ((new_real<<4) & 0xF0) + (new_imaginary & 0x0F);
                        }
                        else{
                            temp_output = ((clipped_offset_default_real<<4) & 0xF0) + (clipped_offset_default_imaginary & 0x0F);
                        }
                        if (XOR_MANUAL){
                            temp_output = temp_output ^ 0x88; //bit flip on sign bit shifts the value by 8: makes unsigned into signed and vice versa.  Currently turns back into signed
                        }
                        packed_data_set[currentAddress] = temp_output;
                    }
                    if (DEBUG_GENERATOR && k == 0)
                        printf("%d ",packed_data_set[currentAddress]);
                }
                if (DEBUG_GENERATOR && k == 0)
                    printf("\n");
            }
        }
    }

    if (DEBUG_GENERATOR)
        printf("END OF DATASET\n");
    return;
}


//int main(int argc, char ** argv){
int main_cpu(int argc, char ** argv){
    double array1[1024];
    //double pi = atan(1)*4;
    int i, index;
    srand(42);
    int n = 8;
    if (argc > 1){
        n = atoi (argv[1]);
        if (n> 512){
            printf("This function only tests to n = 512 complex elements\n");
            return -1;
        }
        if (argc > 2){
            int seed = atoi(argv[2]);
            srand(seed);
        }
    }

    index = 0;
    for(i = 0; i<n; i++){
        array1[index++] = -8;//7.5*sin(i*2*pi/16) - 0.5;//rand()%16-8;
        array1[index++] = -8;//0;//rand()%16-8;
        //printf("%d %d\n",i, ((i<<1)&0x3c)+(i&0x1));
    }

   printf("original data\n");
   for (i = 0; i< n; i++){
       printf("%03d: %f %f\n",i,array1[i*2],array1[i*2+1]);
   }

    //four1(array1, n, 1);
    discreteFourierTransform(array1, n, 1);

   printf("after transform\n");
   for (i = 0; i< n; i++){
       printf("%f %f\n",array1[i*2],array1[i*2+1]);
   }

//    four1(array1, n, -1);

//    printf("after inverse transform\n");
//    for (i = 0; i< n; i++){
//        printf("r %f, i %f\n",array1[i*2],array1[i*2+1]);
//    }
    return 0;
}

int main(int argc, char **argv){
    printf("Using kernel: %s\n", OPENCL_FILENAME_1);

//int main_gpu(int argc, char **argv){
    double elapsed_time;
     //basic setup of CL devices
    cl_int err;

    int dev_number = 0;

    // 1. Get a platform.
    cl_platform_id platform;
    clGetPlatformIDs( 1, &platform, NULL );

    // 2. Find a gpu device.
    cl_device_id deviceID[5];

    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 4, deviceID, NULL);

    if (err != CL_SUCCESS){
        printf("Error getting device IDs\n");
        return (-1);
    }

    cl_ulong lm;
    err = clGetDeviceInfo(deviceID[dev_number], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lm, NULL);
    if (err != CL_SUCCESS){
        printf("Error getting device info\n");
        return (-1);
    }
    //printf("Local Mem: %i\n",lm);

    cl_uint mcl,mcm;
    clGetDeviceInfo(deviceID[dev_number], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &mcl, NULL);
    clGetDeviceInfo(deviceID[dev_number], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &mcm, NULL);
    float card_tflops = mcl*1e6 * mcm*16*4*2 / 1e12;
    printf("Testing on device 0: max %f TFLOPS\n\n", card_tflops);
//     printf ("Theoretical: 5 N log2 (N) = 5 x 64 x 6 = 1920 floating point operations\n%f us/FFT\n", 1920.f*1000000/(mcl*1e6 * mcm*16*4));
    printf ("Theoretical: 5 N log2 (N) = 5 x 128 x 7 = 4480 floating point operations\n%f us/FFT\n", 4480.f*1000000/(mcl*1e6 * mcm*16*4));


    // 3. Create a context and command queues on that device.
    cl_context context = clCreateContext( NULL, 1, &deviceID[dev_number], NULL, NULL, NULL);
    cl_command_queue queue[N_QUEUES];
    for (int i = 0; i < N_QUEUES; i++){
        queue[i] = clCreateCommandQueue( context, deviceID[dev_number], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err );
        //add a more robust error check at some point?
        if (err){ //success returns a 0
            printf("Error initializing queues.  Exiting program.\n");
            return (-1);
        }

    }

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    // 4a load the source files //this load routine is based off of example code in OpenCL in Action by Matthew Scarpino
    char cl_fileNames[3][256];
    sprintf(cl_fileNames[0],OPENCL_FILENAME_1);

    char cl_options[1024]= "-cl-mad-enable";
    //sprintf(cl_options,"-D ACTUAL_NUM_ELEMENTS=%du -D ACTUAL_NUM_FREQUENCIES=%du -D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du -D BASE_TIMESAMPLES_INT_ACCUM=%d", ACTUAL_NUM_ELEM, ACTUAL_NUM_FREQ, NUM_ELEM, NUM_FREQ, num_blocks, NUM_TIMESAMPLES, BASE_TIMESAMPLES_INT_ACCUM);
    size_t cl_programSize[NUM_CL_FILES];
    FILE *fp;
    char *cl_programBuffer[NUM_CL_FILES];


    for (int i = 0; i < NUM_CL_FILES; i++){
        fp = fopen(cl_fileNames[i], "r");
        if (fp == NULL){
            printf("error loading file: %s\n", cl_fileNames[i]);
            return (-1);
        }
        fseek(fp, 0, SEEK_END);
        cl_programSize[i] = ftell(fp);
        rewind(fp);
        cl_programBuffer[i] = (char*)malloc(cl_programSize[i]+1);
        cl_programBuffer[i][cl_programSize[i]] = '\0';
        int sizeRead = fread(cl_programBuffer[i], sizeof(char), cl_programSize[i], fp);
        if (sizeRead < cl_programSize[i])
            printf("Error reading the file!!!");
        fclose(fp);
    }

    cl_program program = clCreateProgramWithSource( context, NUM_CL_FILES, (const char**)cl_programBuffer, cl_programSize, &err );
    if (err){
        printf("Error in clCreateProgramWithSource: %i\n",err);
        return(-1);
    }

    //printf("here1\n");
    size_t log_size;
    char *program_log;
    err = clBuildProgram( program, 1, &deviceID[dev_number], cl_options, NULL, NULL );
    if (err != 0){
        printf("Error in clBuildProgram: %i\n",err);
        clGetProgramBuildInfo (program, deviceID[dev_number], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) calloc(log_size+1, sizeof(char));
        clGetProgramBuildInfo (program, deviceID[dev_number], CL_PROGRAM_BUILD_LOG, log_size+1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        return(-1);
    }

    cl_kernel fft128 = clCreateKernel( program, "upchannelize", &err );
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return (-1);
    }

    for (int i =0; i < NUM_CL_FILES; i++){
        free(cl_programBuffer[i]);
    }

    // 5. set up arrays and initilize if required
    float *array1;
    int total_number_floats;
    int num_data_sets = 8192;
    unsigned char *arrayInput;
    total_number_floats = num_data_sets*2048*2*2;//128*2*num_data_sets;
    array1= (float *) malloc(total_number_floats*sizeof(float));
    if (array1 == NULL){
        printf("Error allocating memory\n");
        exit(-1);
    }
    arrayInput = (unsigned char *) malloc(2048*num_data_sets);  //here input and output are differ by 4)
    if (arrayInput == NULL){
        printf("Error allocating memory\n");
        exit(-1);
    }
    //resultsArray= (float *) malloc(total_number_floats*sizeof(float));
    //if (resultsArray == NULL){
    //    printf("Error allocating memory\n");
    //    exit(-1);
    //}
    //initialize
    int index = 0;
    //for (int j = 0; j < num_data_sets; j++){
    //    srand(42); //seed it and make each of the subsequent arrays equal
    //    for (int i = 0; i < 64; i++){
    //        array1[index++] = rand()%16;
    //        array1[index++] = 0;
    //    }
    //}

    //generate_char_data_set(int generation_Type,
    //                        int random_seed,
    //                        int default_real,
    //                        int default_imaginary,
    //                        int initial_real,
    //                        int initial_imaginary,
    //                        int single_frequency,
    //                        int num_timesteps,
    //                        int num_frequencies,
    //                        int num_elements,
    //                        int num_data_sets,
    //                        unsigned char *packed_data_set)
    generate_char_data_set(GEN_TYPE,GEN_DEFAULT_SEED,GEN_DEFAULT_RE, GEN_DEFAULT_IM,GEN_INITIAL_RE,GEN_INITIAL_IM,GEN_FREQ, num_data_sets, 1, 2048, 1, arrayInput);

    printf("Generated arrayInput (packed, from generate_char_data_set) ------:\n");
    for (int t=0; t < 10; t++){
      printf(" %2d ", arrayInput[t]);
    }
    printf("\n");

    cl_mem device_fourier64data, device_fourier64results, device_hfb_results;
    device_fourier64data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2048*num_data_sets, arrayInput, &err); //note there was a typo previously that USED host memory instead of COPYing host memory.  Speeds are close to theoretical now.
    device_fourier64results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_number_floats * sizeof(float), NULL, &err);
    device_hfb_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_number_floats * sizeof(float), NULL, &err);

    // Set up work sizes
    //int num_data_sets_device = num_data_sets/4; //each work group processes two sets at once
    size_t gws[3]={64*2048/4, 1,num_data_sets/128};
    size_t lws[3]={64, 1, 1};

    //set parameters
    clSetKernelArg(fft128, 0, sizeof (void *), (void *) &device_fourier64data);
    clSetKernelArg(fft128, 1, sizeof (void *), (void *) &device_fourier64results);
    clSetKernelArg(fft128, 2, sizeof (void *), (void *) &device_hfb_results);

    //start the timer
    elapsed_time = e_time();
    //enqueue the kernel
    int extra_coef = 10;
    for (int i = 0; i < extra_coef; i++){
        clEnqueueNDRangeKernel(queue[0],
                            fft128,
                            3, //dimension of GWS/LWS
                            NULL, //no offsets
                            gws,
                            lws,
                            0,
                            NULL,
                            NULL);
    }
    clFinish(queue[0]);
    //stop the timer
    elapsed_time = e_time() - elapsed_time;
    printf ("Computed %d iterations of %d 128 element FFTs in %f s (%f us/FFT on average)\n",extra_coef, num_data_sets*2048/128, elapsed_time, (double)elapsed_time*1000000/num_data_sets/2048*128/extra_coef);

    //read back the result
    err = clEnqueueReadBuffer(queue[0],device_fourier64results, CL_TRUE, 0, total_number_floats*sizeof(float), array1, 0, NULL, NULL);
    //err = clEnqueueReadBuffer(queue[0],device_fourier64results, CL_TRUE, 0, total_number_floats*sizeof(float), array1, 0, NULL, NULL);

    clFinish(queue[0]);

    //compare results with cpu fft
    //calculate fft
    printf("total_number_floats: %d\n", total_number_floats);
    double *array2;
    array2 = (double *) malloc(total_number_floats*sizeof(double));
    if (array2 == NULL){
        printf("Error allocating memory");
        exit (-1);
    }
    printf("created array: array2\n");
    //'transpose' the array (elements <-> time) while expanding it out for testing with cpu fft routines
    //(this is done by swapping the two indice for the nested loop--it would ordinarily have j on the outer loop)
    index = 0;
    for (int i = 0; i < 2048; i++){
        for (int j = 0; j < num_data_sets/128; j++){
            array2[index++] = HI_NIBBLE(arrayInput[j*2048+i])-8; // to give a range from -8 to 7
            array2[index++] = LO_NIBBLE(arrayInput[j*2048+i])-8; // to give a range from -8 to 7;
            //if (j == 0)
            //    printf("%4d %6.1f %6.1f\n",i,array2[index-2],array2[index-1]);
        }
    }

    printf("This is array2 as allocated with HI-/LO-NIBBLE - tranposed\n");
    for (int i = 0; i < 5; i++){
      printf("(%.2f %.2f) ",array2[i*2], array2[i*2+1]);
    }
    printf("\n next 2048\n");
    for (int i = 2048; i < 2053; i++){
      printf("(%.2f %.2f) ",array2[i*2], array2[i*2+1]);
    }

    printf("\npre-fft; num_data_sets*2048/128 = %d\n", num_data_sets*2048/128);
    //now calculate the fft of the data sets (num_data_sets referred to number of overall timesteps, more or less).
    for (int j = 0; j < num_data_sets*2048/128; j++){
        four1double(&array2[j*128*2],128, 1);
    }
    printf("fft finished\n");

    //need to reorder the data from the FFT or redo the comparison.
    //presently has order: [Elem0, Freq0-127(upchannelized), Time0],Time1, ..., Time(num_data_sets/128), Elem1...
    //need it to go to Elem fastest varying, then Freq, then Time
    //create a new array that is twice as large (elements above 256 are zero padding)
    double *padded_array;
    index = 0;
    padded_array = (double *) malloc(total_number_floats*2*sizeof(double));
    if (padded_array == NULL){
        printf("Error allocating memory");
        exit (-1);
    }
    printf("created empty array to store padded and sorted data\n");
    for (int k = 0; k < num_data_sets/128; k++){ //time loop
        for (int j = 0; j < 128; j++){
            for (int i = 0; i < 2048; i++){
                padded_array[index++]=array2[2*(i*num_data_sets + k*128 + j)];
                padded_array[index++]=array2[2*(i*num_data_sets + k*128 + j)+1];
                //padding for spatial transform every 256 items, since there are 256 elements (per polarization) in a line in each Full CHIME cylinder
                if ((i+1)%256 == 0){
                    for (int ii = 0; ii < 256; ii++){
                        padded_array[index++] = 0;
                        padded_array[index++] = 0;
                    }
                }
            }

        }
    }

    printf("sorting and padding complete\n");
    /*
    for (int j = 0; j < 129; j++){
        for (int i = 0; i < 4096; i++){
            printf("%4d %4d GPU: %13.7f %13.7f\n", j, i, array1[j*4096*2+ i*2],array1[j*4096*2+ i*2+1]);
        }
	}*/
//    for (int i = 0; i < 4096; i++){
//       printf("%4d: %13.8f %13.8f %13.8f %13.8f, %13.8f %13.8f\n", i, RelativeAndAbsdouble(array1[i*2],padded_array[i*2], FLT_EPSILON), RelativeAndAbsdouble(array1[i*2+1],padded_array[i*2+1], FLT_EPSILON),array1[i*2],padded_array[i*2], array1[i*2+1],padded_array[i*2+1]);
//    }

    float eps = FLT_EPSILON;
    printf("FLT_EPSILON: %1.12f\n\n", eps);
    free(array1);
    free(arrayInput);
    free(array2);
    free(padded_array);

    clReleaseKernel(fft128);
    clReleaseProgram(program);
    clReleaseMemObject(device_fourier64data);
    clReleaseMemObject(device_fourier64results);
    clReleaseCommandQueue(queue[0]);
    clReleaseContext(context);
    return 0;
}

