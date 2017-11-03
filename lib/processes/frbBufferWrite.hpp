#ifndef FRBBUFFERWRITE_HPP
#define FRBBUFFERWRITE_HPP
 
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <string>
 
class frbBufferWrite : public KotekanProcess {
public:
  frbBufferWrite(Config& config,
  const string& unique_name,
  bufferContainer &buffer_container);
  virtual ~frbBufferWrite();
  void apply_config(uint64_t fpga_seq) override;
  void main_thread();
private:
  struct Buffer *frb_buf;
  
};
 
#endif

