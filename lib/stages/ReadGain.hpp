
#ifndef READ_GAIN
#define READ_GAIN

#include "Stage.hpp"
#include "restServer.hpp"

#include <vector>

using std::vector;

class ReadGain : public kotekan::Stage {
public:
  /// Constructor.
  ReadGain(kotekan::Config& config_, const string& unique_name,
	   kotekan::bufferContainer& buffer_container);

  void main_thread() override;

  bool update_gains_callback(nlohmann::json& json);


private:

    struct Buffer* gain_buf;
  
    /// Directory path where gain files are
    string _gain_dir;
    /// Default gain values if gain file is missing for this freq, currently set to 1+1j
    vector<float> default_gains;
  
    /// Buffer for accessing metadata
    Buffer* metadata_buf;
    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Metadata buffer precondition ID
    int32_t metadata_buffer_precondition_id;

  /// Freq bin index, where the 0th is at 800MHz
    int32_t freq_idx;
    /// Freq in MHz
    float freq_MHz;

      /// Scaling factor to be applied on the gains, currently set to 1.0 and somewhat deprecated?
    float scaling;

    /// Flag to control gains to be only loaded on request.
    bool update_gains;

    /// Number of elements, should be 2048
    uint32_t _num_elements;

};


#endif
