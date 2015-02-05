//Box-Muller Gaussian Distribution.

//polar form of Box-Muller form
//Using uniformly distributed random numbers returns a value with mean, mu, and standard dev, sigma
//keeps the second val for next time (but therefore is not thread safe)
int normal_dist_Box_Muller(double mu, double sigma){
    static double second_value = 0.0;
    int out_value;
    double rand_val;
    if (second_value == 0.0){
        double u, v, length;
        do {
            u = sfmt_genrand_res53(&sfmt)*2.0 -1.0;
            v = sfmt_genrand_res53(&sfmt)*2.0 -1.0;
            length = u*u+v*v;
        } while (length == 0 || length >= 1.0);
        double coeff = sqrt(-2.0*log(length)/length);
        rand_val = coeff*v*sigma + mu;
        second_value = coeff*u;
        return (int)(round(rand_val));
    }
    else{
        rand_val = second_value*sigma + mu;
        second_value = 0.0;
        return (int)(round(rand_val));
    }
}