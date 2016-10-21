#ifndef RAW_FILE_WRITE_H
#define RAW_FILE_WRITE_H

#include "buffers.h"
#include "KotekanProcess.hpp"

#define RAW_FILE_STR_LEN 64

class rawFileWrite : public KotekanProcess {
public:
    rawFileWrite(Config &config,
                 struct Buffer &buf,
                 int disk_id,
                 char * extension,
                 int write_data,
                 char * dataset_name);
    virtual ~rawFileWrite();
    void main_thread();
private:
    struct Buffer &buf;
    int disk_id;
    char extension[RAW_FILE_STR_LEN];
    int write_data;
    char dataset_name[RAW_FILE_STR_LEN];
};

#endif
