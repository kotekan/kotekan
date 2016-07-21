#ifndef SAMPLEPROCESS_H
#define SAMPLEPROCESS_H

#include "KotekanProcess.h"

class SampleProcess : public KotekanProcess {
public:
    SampleProcess(struct Config &config);
    virtual ~SampleProcess();
    void main_thread() override;
private:

};

#endif /* SAMPLEPROCESS_H */

