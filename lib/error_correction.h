/// @file error_correction.h
/// When correlating in the GPU errors are simply ignored for efficiency,
/// this set of functions deals with tracking those errors, and normalizing
/// the visibilities once the GPU is finished processing them.

#ifndef ERROR_CORRECTION
#define ERROR_CORRECTION

#include <error.h>

#define complex _Complex


#ifdef __cplusplus
   #define complex _Complex
   extern "C" {
#else
   #include <complex.h>
#endif



struct ErrorMatrix {

    int num_freq;
    int num_elements;

    // Number of fully currupt timesamples
    // Used in cases when there is packet loss, or too
    // many errors in the packet to be dealt with.
    int bad_timesamples;

    // An array of length elements*freq, with an element-major encoding,
    // tracking the number of errors that each element has encountered in each
    // frequency.
    int * element_error_counts;

    // An array of length ( elements*(elements + 1)/2 ) * freq,
    // containing the number of times each visibility has been zeroed.
    // This array is used to normalize the visibility matrix once it has been
    // integrated for a while.
    int * correction_factors;
};

void initalize_error_matrix(struct ErrorMatrix* error_matrix, const int num_freq,
                         const int num_elements);

void delete_error_matrix(struct ErrorMatrix * error_matrix);

void reset_error_matrix(struct ErrorMatrix* error_matrix);

void add_bad_timesamples(struct ErrorMatrix* error_matrix, int num_bad_timesamples);

void add_errors(struct ErrorMatrix* error_matrix, int freq, int * new_errors, const int len);

void finalize_error_matrix(struct ErrorMatrix* error_matrix);

// Currently assumes the visibilities are ordered in the natural way.
void apply_error_corrections(struct ErrorMatrix* error_matrix,
                             double complex * visibilities, int num_steps);

#ifdef __cplusplus
}
#endif

#endif