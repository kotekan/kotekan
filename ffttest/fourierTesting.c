// gcc -std=gnu99 -O3 -Wall fourierTesting.c -I$AMDAPPSDKROOT/include -lOpenCL -lm -L$AMDAPPSDKROOT/lib/x86_64/ -o fourierTest

#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include <sys/time.h>
#define OPENCL_FILENAME_1   "fft64b2.cl"
#define NUM_CL_FILES        1
#define N_QUEUES            1

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

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

double e_time(void){
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

void fourier16complex(float *data, int sign){

}

//int main(int argc, char ** argv){
int main_cpu(int argc, char ** argv){
    float array1[1024];
    int i, index;
    srand(42);
    int n = 8;
    if (argc > 1){
        n = atoi (argv[1]);
        if (n> 512){
            printf("This function only tests to n = 512 complex elements\n");
            return -1;
        }
    }

    index = 0;
    for(i = 0; i<n; i++){
        array1[index++] = rand()%32;
        array1[index++] = 0;
        //printf("%d %d\n",i, ((i<<1)&0x3c)+(i&0x1));
    }

   printf("original data\n");
   for (i = 0; i< n; i++){
       printf("%03d: r %f, i %f\n",i,array1[i*2],array1[i*2+1]);
   }

    four1(array1, n, 1);

   printf("after transform\n");
   for (i = 0; i< n; i++){
       printf("r %f, i %f\n",array1[i*2],array1[i*2+1]);
   }

//    four1(array1, n, -1);

//    printf("after inverse transform\n");
//    for (i = 0; i< n; i++){
//        printf("r %f, i %f\n",array1[i*2],array1[i*2+1]);
//    }
    return 0;
}

int main(int argc, char **argv){
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
    printf ("Theoretical: 5 N log2 (N) = 5 x 64 x 6 = 1920 floating point operations\n%f us/FFT\n", 1920.f*1000000/(mcl*1e6 * mcm*16*4));


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

    cl_kernel fft64 = clCreateKernel( program, "FFT64", &err );
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return (-1);
    }

    for (int i =0; i < NUM_CL_FILES; i++){
        free(cl_programBuffer[i]);
    }

    // 5. set up arrays and initilize if required
    float *array1; // make the array larger than 128 (2 x 64 for real and complex) to test how it does for higher multiple addresses
    int total_number_floats;
    int num_data_sets = 500000;
    total_number_floats = 64*2*num_data_sets;
    array1= (float *) malloc(total_number_floats*sizeof(float));
    if (array1 == NULL){
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
    for (int j = 0; j < num_data_sets; j++){
        srand(42); //seed it and make each of the subsequent arrays equal
        for (int i = 0; i < 64; i++){
            array1[index++] = rand()%32;
            array1[index++] = rand()%32;
        }
    }

    cl_mem device_fourier64data, device_fourier64results;
    device_fourier64data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total_number_floats*sizeof(float), array1, &err); //note there was a typo previously that USED host memory instead of COPYing host memory.  Speeds are close to theoretical now.
    device_fourier64results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_number_floats * sizeof(float), NULL, &err);

    // Set up work sizes
    int num_data_sets_device = num_data_sets/2; //each work group processes two sets at once
    size_t gws[3]={64*num_data_sets_device, 1,1};
    size_t lws[3]={64, 1, 1};

    //set parameters
    int sign = 1;
    clSetKernelArg(fft64, 0, sizeof (void *), (void *) &device_fourier64data);
    clSetKernelArg(fft64, 1, sizeof (int), &sign);
    clSetKernelArg(fft64, 2, sizeof (void *), (void *) &device_fourier64results);

    //start the timer
    elapsed_time = e_time();
    //enqueue the kernel
    int extra_coef = 1000;
    for (int i = 0; i < extra_coef; i++){
        clEnqueueNDRangeKernel(queue[0],
                            fft64,
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
    printf ("Computed %d 64 element FFTs in %f s (%f us/FFT on average)\n",num_data_sets, elapsed_time, (double)elapsed_time*1000000/num_data_sets/extra_coef);
    //     sign = -1;
//     clSetKernelArg(fft64, 1, sizeof (int), &sign);
//
//     //enqueue the kernel
//     clEnqueueNDRangeKernel(queue[0],
//                            fft64,
//                            3, //dimension of GWS/LWS
//                            NULL, //no offsets
//                            gws,
//                            lws,
//                            0,
//                            NULL,
//                            NULL);
//
//     clFinish(queue[0]);


    //read back the result
    err = clEnqueueReadBuffer(queue[0],device_fourier64results, CL_TRUE, 0, total_number_floats*sizeof(float), array1, 0, NULL, NULL);
    //printf("err: %d\n",err);

    clFinish(queue[0]);

    //compare results with cpu fft
    //calculate fft
    float array2 [128];
    srand(42); //seed it and make each of the subsequent arrays equal
    index = 0;
    for (int i = 0; i < 64; i++){
        array2[index++] = rand()%32;
        array2[index++] = rand()%32;
    }
    four1(array2, 64, 1);
    //

    float eps = 0.00005;
    int error_count = 0;
    for (int j = 0; j < num_data_sets; j++){
        //printf("j: %d \n",j);
        for (int i = 0; i < 64; i++){
            if (j == 0)
                printf("i: %d, RE: %f, IM: %f\n", i, array1[j*64*2+i*2], array1[j*64*2+i*2+1]);
            if (fabs(array1[j*64*2+i*2] - array2[i*2]) > eps || fabs(array1[j*64*2+i*2+1]- array2[i*2+1]) > eps){
                //printf("j: %d i: %d\n",j, i);
                error_count++;
            }
        }
    }
    printf("#errors: %d\n", error_count);
    free(array1);
    //free(resultsArray);

    clReleaseKernel(fft64);
    clReleaseProgram(program);
    clReleaseMemObject(device_fourier64data);
    clReleaseMemObject(device_fourier64results);
    clReleaseCommandQueue(queue[0]);
    clReleaseContext(context);
    return 0;
}
