
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include "errors.h"
#include "error_correction.h"

void initalize_error_matrix(struct ErrorMatrix* error_matrix, const int num_freq, const int num_elements)
{

    error_matrix->num_elements = num_elements;
    error_matrix->num_freq = num_freq;

    error_matrix->bad_frames = malloc(num_freq * sizeof(int));
    CHECK_MEM(error_matrix->bad_frames);

    error_matrix->element_error_counts = malloc(num_elements * num_freq * sizeof(int));
    CHECK_MEM(error_matrix->element_error_counts);

    error_matrix->correction_factors = malloc((num_elements*(num_elements + 1)/2) * 
                                                                num_freq * sizeof(int));
    CHECK_MEM(error_matrix->correction_factors);
}


void reset_error_matrix(struct ErrorMatrix* error_matrix)
{
    memset(error_matrix->bad_frames, 0, error_matrix->num_freq);
    memset(error_matrix->element_error_counts, 0, error_matrix->num_elements * error_matrix->num_freq);
    memset(error_matrix->correction_factors, 0, (error_matrix->num_elements*(error_matrix->num_elements + 1)/2) * error_matrix->num_freq);
}


void delete_error_matrix(struct ErrorMatrix* error_matrix)
{
    free(error_matrix->bad_frames);
    free(error_matrix->correction_factors);
    free(error_matrix->element_error_counts);
}


void add_errors(struct ErrorMatrix* error_matrix, int freq, 
                int* new_errors, const int len)
{
    int index = 0; // Position in the linear array.
    int r = 0; // Row
    int c = 0; // Column
    int n = error_matrix->num_elements; // Dimension of the matrix.

    // This algorithm is O(N^2), so len must be fairly small for it to be
    // computed on the CPU.
    for (int i = 0; i < len; ++i) {

        // Add to the number of errors for the given element and freq.
        error_matrix->element_error_counts[freq * error_matrix->num_elements + new_errors[i]] += 1;

        // We don't need to check all n^2 pairs, since we can just reflect about the
        // diagonal any values that fall in the lower triangle.  This can be shown to
        // cover all the pairs we require.
        for (int j = i; j < len; ++j) {
            // We are on a diagonal, or in the upper triangle.
            if (new_errors[i] <= new_errors[j]) {
                r = new_errors[i];
                c = new_errors[j];
            } else {
                // The index is in the lower triangle, so we reflect about the diagonal.
                // (swap the indices)
                r = new_errors[j];
                c = new_errors[i];
            }
            index = freq * ((n*(n + 1)) / 2)
                        + r + n*c - ((c*(c + 1)) / 2);
            error_matrix->correction_factors[index] -= 1;
        }
    }
}

void finalize_error_matrix(struct ErrorMatrix* error_matrix)
{
    int index = 0;
    // Cover all frequencies.
    for (int k = 0; k < error_matrix->num_freq; ++k) {

        // This is a full O(n^2) operation, but it only needs to be
        // done once per integration period.
        for (int i = 0; i < error_matrix->num_elements; ++i) {
            for (int j = i; j < error_matrix->num_elements; ++j) {

                // Add errors for each time a frame was discarded/lost.
                error_matrix->correction_factors[index] += error_matrix->bad_frames[k];

                // Add errors on elements. Note: we double add here in some cases, but 
                // these are corrected by the existing offsets created when the error was added.
                error_matrix->correction_factors[index] += error_matrix->element_error_counts[i];
                error_matrix->correction_factors[index] += error_matrix->element_error_counts[j];

                index++;
            }
        }
    }
}

void add_bad_frames(struct ErrorMatrix* error_matrix, int freq, int num_bad_frames)
{
    assert(error_matrix != NULL);
    assert(freq <= error_matrix->num_freq);

    error_matrix->bad_frames[freq] += 1;
}

void apply_error_corrections(struct ErrorMatrix* error_matrix, 
                             double complex * visibilities, int num_steps)
{

    int n = ((error_matrix->num_elements * (error_matrix->num_elements + 1)) / 2) 
                * error_matrix->num_elements;

    for (int i = 0; i < n; ++i) {
        visibilities[i] = visibilities[i] * (double)error_matrix->correction_factors[i] / 
                            ((double)num_steps - (double)error_matrix->correction_factors[i]);
    }
}




