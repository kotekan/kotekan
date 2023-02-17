#ifndef ONEHOT_METADATA
#define ONEHOT_METADATA

#include "Telescope.hpp"
#include "buffer.h"
#include "chimeMetadata.hpp"
#include "datasetManager.hpp"
#include "metadata.h"

#include <sys/time.h>
#include <vector>

#pragma pack()

struct oneHotMetadata {
    struct chimeMetadata chime;
    int frame_counter;
    // TODO -- make this a static array because metadata objects get c-copied around.
    std::vector<int> hotIndices;
};

inline bool metadata_is_onehot(struct Buffer* buf, int frame_id) {
    // avoid gcc warning about being unused.
    (void)frame_id;
    return strcmp(buf->metadata_pool->type_name, "oneHotMetadata") == 0;
}

inline void set_onehot_frame_counter(struct Buffer* buf, int frame_id, int counter) {
    struct oneHotMetadata* m = (struct oneHotMetadata*)buf->metadata[frame_id]->metadata;
    m->frame_counter = counter;
}

inline int get_onehot_frame_counter(struct Buffer* buf, int frame_id) {
    struct oneHotMetadata* m = (struct oneHotMetadata*)buf->metadata[frame_id]->metadata;
    return m->frame_counter;
}

inline void set_onehot_indices(struct Buffer* buf, int frame_id, std::vector<int> indices) {
    struct oneHotMetadata* m = (struct oneHotMetadata*)buf->metadata[frame_id]->metadata;
    m->hotIndices = indices;
}

inline std::vector<int> get_onehot_indices(struct Buffer* buf, int frame_id) {
    struct oneHotMetadata* m = (struct oneHotMetadata*)buf->metadata[frame_id]->metadata;
    return m->hotIndices;
}

#endif
