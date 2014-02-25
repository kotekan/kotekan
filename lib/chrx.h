//! \file chrx.h
//! Header file for python module chrx.h
//==============================================================================

#ifndef CHRX_H
#define CHRX_H

#include <python2.7/Python.h>
#include <complex.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <pthread.h>


//==============================================================================
//==============================================================================
// Definitions for the acquisition frame.
//==============================================================================
//==============================================================================

//! The maximum length of a name in an HDF5 file.
#define HDF5_NAME_LEN   65

// The prototype will be defined below; we need other structures to know that
// it exists meanwhile.
struct chrx_acq_t;

//! A complex integer datatype.
typedef struct {
  int32_t real; //!< The real component.
  int32_t imag; //!< The imaginary component.
} complex_int_t;

//! A structure for holding flags related to the visibilities.
struct vis_flag_t {
  uint8_t fpga;     //!< Flags from the FPGA.
  uint8_t rfi;      //!< Flags from RFI filtering.
};

//! A structure for defining the visibility data layout.
struct vis_layout_t {
  uint16_t ant_chan_a;      //!< Antenna channel A.
  uint16_t ant_chan_b;      //!< Antenna channel B.
};

//! The structure for holding timestamps.
struct timestamp_t {
  unsigned int fpga_count;             //!< The FPGA counter number.
  unsigned int cpu_s;                  //!< The CPU time (seconds).
  unsigned int cpu_us;                 //!< The CPU time (microseconds).
};

//! The data that gets written to disc.
struct disc_frame_t {
  struct timestamp_t *timestamp;    //!< The timestamps.
  complex_int_t *vis;               //!< The visibilities.
  struct vis_flag_t *vis_flag;      //!< The visibliity flags.
  float *fpga_hk;                   //!< FPGA ambient temperatures.
  int16_t *serial_adc;              //!< Serial ADC values.
};

//! This structure maps an HDF5 table to the correct offset in #disc_frame_t.
struct hdf5_table_map_t {
  char name[HDF5_NAME_LEN]; //!< The name of the table for the HDF5 file.
  size_t pos;               //!< The position of the data in #disc_frame_t.
  //! A function for creating the HDF5 table.
  hid_t (*init_func)(struct chrx_acq_t *, const char *, hid_t);
  hid_t tab;                //!< The HDF5 handle to the table.
};

//! A macro for filling the hdf5_table_map_t struct.
//! It assumes a standardised naming convention, where "name" appears in:
//! - The #hdf5_table_map_t struct.
//! - A function of the form <tt>hid_t new_hdf5_table_<name>(hid_t fd)</tt>
#define HTAB_ENTRY(name) {#name, offsetof(struct disc_frame_t, name), \
                          new_hdf5_table_##name, -1}
//! For terminating the #hdf5_table_map_t array.
#define HTAB_ENTRY_END  {"", 0, NULL, -1}

//! A structure for mapping serial ADC input to disc.
struct ser_adc_frame_map_t { 
  char name[HDF5_NAME_LEN]; //!< The name of the serial ADC channel.
  int adc;                  //!< The index of the ADC to read.
  char cal[HDF5_NAME_LEN];  //!< The calibration being used.
};

hid_t new_hdf5_table_timestamp(struct chrx_acq_t *self, const char *name,
                               hid_t fd);
hid_t new_hdf5_table_vis(struct chrx_acq_t *self, const char *name, hid_t fd);
hid_t new_hdf5_table_vis_flag(struct chrx_acq_t *self, const char *name,
                              hid_t fd);
hid_t new_hdf5_table_fpga_hk(struct chrx_acq_t *self, const char *name,
                             hid_t fd);
hid_t new_hdf5_table_serial_adc(struct chrx_acq_t *self, const char *name,
                                hid_t fd);

int get_frame_page(struct chrx_acq_t *self);
int map_ser_adc_to_frame(struct chrx_acq_t *self);




//==============================================================================
//==============================================================================
// Some function prototypes.
//==============================================================================
//==============================================================================

//! For logging.
enum mlog_level_t {
  M_INFO = 20,
  M_DEBUG = 10,
  M_WARN = 30,
  M_ERR = 40,
  M_CRIT = 50
};

void *fpga_thread(void *arg);
void *ser_adc_thread(void *arg);
void *disc_thread(void *arg);
void mlog_full(struct chrx_acq_t *self, enum mlog_level_t mlog_level,
          const char *format, ...);
#define mlog(l, fmt, ...) mlog_full(self, l, fmt, ##__VA_ARGS__)
PyObject *grab_conf_val(PyObject *conf, const char *key);
PyObject *grab_conf_dict(struct chrx_acq_t *self, const char *key);




//==============================================================================
//==============================================================================
// Definitions for the chrx python module.
//==============================================================================
//==============================================================================

//! The Python chrx module.
struct chrx_t {
  PyObject_HEAD     //!< Default python object definition.
};

//! For storing integer header information.
struct chrx_acq_head_int_t {
  char type;    //!< Should be set to 'i'.
  char *name;   //!< The name of the header item.
  int *val;     //!< The value(s) of the header item.
  int n_val;    //!< The number of values.
};

//! For storing double header information.
struct chrx_acq_head_double_t {
  char type;    //!< Should be set to 'd'.
  char *name;   //!< The name of the header item.
  double *val;  //!< The value(s) of the header item.
  int n_val;    //!< The number of values.
};

//! For storing string header information.
struct chrx_acq_head_string_t {
  char type;    //!< Should be set to 's'.
  char *name;   //!< The name of the header item.
  char **val;   //!< The value(s) of the header item.
  int n_val;    //!< The number of values.
};

//! For storing header information.
union chrx_acq_head_t {
  struct chrx_acq_head_int_t i;
  struct chrx_acq_head_double_t d;
  struct chrx_acq_head_string_t s;
};

//! The Python data acquisition class.
struct chrx_acq_t {
  // Python stuff.
  PyObject_HEAD     //!< Default python object definition.

  // Variables used by all acquisition systems.
  PyObject *conf;   //!< The Python configuration object.
  PyObject *log;
  //! The prefix for all data files (e.g., /data/[timestamp]/[timestamp].h5).
  char *path_prefix;
  int running;      //!< True if acquisition is running, false otherwise.
  int n_freq;       //!< The number of frequency bins.
  int n_ant;        //!< The number of antenna channels.
  int n_corr;       //!< The number of correlations == n_chan * (n_chan + 1) / 2

  // FPGA acquisition variables.
  pthread_t fpga_loop;  //!< The variable for the FPGA listening thread.
  uint8_t **udp_buf;    //!< A circular buffer for storing UDP packets.
  double complex *vis_sum; //!< For averaging the visibilities from UDP packets.
  int *udp_buf_len;     //!< For storing the length of the packets in udp_buf.
  int udp_port;     //!< The IP port number for listening to the FPGA.
  int n_adc;        //!< The number of FPGA units.
  int udp_max_len;  //!< Maximum length of UDP packets from the FPGA, in bytes.
  int udp_n_buf;    //!< Number of buffers in udp_buf.
  int udp_w_pos;    //!< Write position of the circular udp_buf.
  int udp_r_pos;    //!< Read position of the circular udp_buf.

  // Serial ADC housekeeping.
  pthread_t ser_adc_loop;   //!< The variable for the FPGA housekeeping thread.
  char *ser_dev;            //!< The device name for the ADC serial device.
  int fd_ser;               //!< File descriptor for the ADC serial device.
  int n_ser_adc;            //!< Number of serial ADC channels.
  //! Serial ADC frame information.
  struct ser_adc_frame_map_t *ser_adc_frame_map;
  double ser_timeout;       //!< Number of seconds to wait for serial read.

  // Disc frame data.
  pthread_t disc_loop;          //!< The variable for the disc-writing thread.
  pthread_mutex_t frame_lock;   //!< For locking disc-writing variables.
  pthread_mutex_t frame_lock2;   //!< For locking disc-writing variables.
  struct disc_frame_t *frame;   //!< Frames to be written to disc.
  int n_frame_page;             //!< Number of frame pages.
  //! Which frame page in the frame buffer is currently being filled.
  int frame_page;
  //! Number of FPGA samples that get averaged into a single disc frame.
  int udp_spf;
  int n_fpga_sample;            //!< Number of FPGA samples currently averaged.
  int frame_per_file;           //!< Number of frames per HDF5 file.

  // HDF5 variables.
  hid_t h_fd;           //!< The HDF5 file descriptor.
  hid_t hg_cal;         //!< The HDF5 group for calibration information.

  // Acquisition header data.
  union chrx_acq_head_t *head;  //!< Acquisition header information.
  int n_head;                   //!< Number of header items (length of head).
};

//! The Python data acquisition object.
extern PyTypeObject chrx_acq;

int chrx_acq_init(struct chrx_acq_t *self);
void chrx_acq_delete(struct chrx_acq_t *self);


//! A macro for defining Python variable get functions.
#define GET_FUNCTION(var, pyvar_type, cvar_type) \
  static PyObject *chrx_acq_get_##var(struct chrx_acq_t *self, \
                                       void *closure) { \
    return Py##pyvar_type##_From##cvar_type(self->var); \
  }

#define SET_FUNCTION(var, pyvar_type, cvar_type) \
  static int chrx_acq_set_##var(struct chrx_acq_t *self,\
                                     PyObject *val, void *closure) { \
    if (!Py##pyvar_type##_Check(val)) { \
      PyErr_SetString(PyExc_TypeError, "'chrx.acq' attribute 'val' must " \
                                       "be of type " #pyvar_type); \
      return -1; \
    } \
    self->var = Py##pyvar_type##_As##cvar_type(val); \
    return 0; \
  }

#define GETSET_FUNCTIONS(type, memb, var, pyvar_type, cvar_type) \
  GET_FUNCTION(type, memb, var, pyvar_type, cvar_type) \
  SET_FUNCTION(type, memb, var, pyvar_type, cvar_type)

//! A macro for defining Python variable get/set functions, for strings.
#define GETSET_STR_FUNCTIONS(var) \
  static PyObject *chrx_acq_get_##var(struct chrx_acq_t *self, void *closure) { \
    return PyString_FromString(self->var); \
  } \
  static int chrx_acq_set_##var(struct chrx_acq_t *self, PyObject *val, \
                               void *closure) { \
    if (!PyString_Check(val)) { \
      PyErr_SetString(PyExc_TypeError, "'chrx.acq' attribute 'val' must " \
                                       "be of type String"); \
      return -1; \
    } \
    strcpy(self->var, PyString_AsString(val)); \
    return 0; \
  }

//! A macro for Python get/set entries in the PyGetSetDef array.
#define GETSET_STRUCT(var, docstring) \
  {#var, (getter)chrx_acq_get_##var, (setter)chrx_acq_set_##var, \
   docstring},

//! A macro for Python get entries in the PyGetSetDef array.
#define GET_STRUCT(var, docstring) \
  {#var, (getter)chrx_acq_get_##var, NULL, \
   docstring},

#endif
