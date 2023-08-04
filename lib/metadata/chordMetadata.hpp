#ifndef CHORD_METADATA
#define CHORD_METADATA

#include "Telescope.hpp"
#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "datasetManager.hpp"
#include "metadata.h"

#include <sys/time.h>
#include <vector>

#pragma pack()

struct chordMetadata {
    struct chimeMetadata chime;
    int frame_counter;
    // TODO -- make this a static array because metadata objects get c-copied around.
    //std::vector<int> hotIndices;
};

inline void chord_metadata_copy(struct chordMetadata* out, const struct chordMetadata* in) {
    memcpy(out, in, sizeof(struct chordMetadata));
}

#endif
