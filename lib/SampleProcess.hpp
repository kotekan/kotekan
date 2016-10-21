#ifndef SAMPLEPROCESS_H
#define SAMPLEPROCESS_H

#include "KotekanProcess.hpp"

class SampleProcess : public KotekanProcess {
public:
    SampleProcess(Config &config);
    virtual ~SampleProcess();
    void main_thread();
private:

};

#endif /* SAMPLEPROCESS_H */

