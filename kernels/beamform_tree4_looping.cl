#define TIME_SCL 32

__kernel void gpu_beamforming(const int n_freq,
                              const int n_elem,
                              __constant float *phases,
                              __local float* lds_data,
                              __global unsigned int *data,
                              __global int *output
                            )
{
    //calculate the frequency for this kernel, in GHz
    const float freq = 0.4*(get_global_id(1)/((float)n_freq) + 1);  //NEEDS TO BE CHECKED/FIXED LATER+++++++++++++++++
    float R[4], I[4];
    float outR, outI;

    //calculate the phase correction
    float phase_re[4], phase_im[4];
    phase_im[0] = sincos(phases[get_global_id(0)*4+0]*freq, &phase_re[0]);
    phase_im[1] = sincos(phases[get_global_id(0)*4+1]*freq, &phase_re[1]);
    phase_im[2] = sincos(phases[get_global_id(0)*4+2]*freq, &phase_re[2]);
    phase_im[3] = sincos(phases[get_global_id(0)*4+3]*freq, &phase_re[3]);

    for (uint t=0; t<TIME_SCL; t++)
    {
        // read in the data
        uint readData = data[get_global_id(0) + (n_elem/4) * 
                              (get_global_id(1) + n_freq*(get_global_id(2)*TIME_SCL+t)) ];

        I[0] = ((float)((readData    )&0xF)) - 8;
        I[1] = ((float)((readData>> 8)&0xF)) - 8;
        I[2] = ((float)((readData>>16)&0xF)) - 8;
        I[3] = ((float)((readData>>24)&0xF)) - 8;

        R[0] = ((float)((readData>> 4)&0xF)) - 8;
        R[1] = ((float)((readData>>12)&0xF)) - 8;
        R[2] = ((float)((readData>>20)&0xF)) - 8;
        R[3] = ((float)((readData>>28)&0xF)) - 8;

        outR = R[0]*phase_re[0] + I[0]*phase_im[0] +
               R[1]*phase_re[1] + I[1]*phase_im[1] +
               R[2]*phase_re[2] + I[2]*phase_im[2] +
               R[3]*phase_re[3] + I[3]*phase_im[3];
        outI = I[0]*phase_re[0] - R[0]*phase_im[0] +
               I[1]*phase_re[1] - R[1]*phase_im[1] +
               I[2]*phase_re[2] - R[2]*phase_im[2] +
               I[3]*phase_re[3] - R[3]*phase_im[3];

        barrier(CLK_LOCAL_MEM_FENCE);
        lds_data[get_local_id(0)         ] = outR;
        lds_data[get_local_id(0)+n_elem/4] = outI;

        //tree sum: each kernel adds four and flops them back into local share
        //i = number of elements left to sum; should be divide-by-2 or -by-4
        for (uint i=n_elem/4; i>1; i/=4){
            barrier(CLK_LOCAL_MEM_FENCE);
            if (get_local_id(0) >= i/4) continue;

            R[0]=lds_data[get_local_id(0)*4+0];
            R[1]=lds_data[get_local_id(0)*4+1];
            R[2]=lds_data[get_local_id(0)*4+2];
            R[3]=lds_data[get_local_id(0)*4+3];

            I[0]=lds_data[get_local_id(0)*4+i+0];
            I[1]=lds_data[get_local_id(0)*4+i+1];
            I[2]=lds_data[get_local_id(0)*4+i+2];
            I[3]=lds_data[get_local_id(0)*4+i+3];

            barrier(CLK_LOCAL_MEM_FENCE);
            outR = R[0]+R[1]+R[2]+R[3];
            outI = I[0]+I[1]+I[2]+I[3];
            lds_data[get_local_id(0)    ]=outR;
            lds_data[get_local_id(0)+i/4]=outI;
        }

        // write output to buffer as an int, shift 16 bits up
        if (get_local_id(0) == 0) {
            output[2*(get_global_id(1)+ (get_global_id(2)*TIME_SCL+t)*n_freq)  ] = (int)(lds_data[0]*(1<<16));
            output[2*(get_global_id(1)+ (get_global_id(2)*TIME_SCL+t)*n_freq)+1] = (int)(lds_data[1]*(1<<16));
        }
    }

    return; 
}
