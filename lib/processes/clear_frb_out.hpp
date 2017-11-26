#ifndef CLEARFRBOUT_HPP
#define CLEARFRBOUT_HPP
 
#include "buffer.h"
#include "KotekanProcess.hpp"
#include <string>
 
class clear_frb_out : public KotekanProcess {
public:
  clear_frb_out(Config& config,
  const string& unique_name,
  bufferContainer &buffer_container);
  virtual ~clear_frb_out();
  void apply_config(uint64_t fpga_seq) override;
  void main_thread();
private:
  struct Buffer *frb_buf;
};
 
#endif

