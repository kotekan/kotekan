//! \file frame.c
//! Functions for frame read/write control.
//==============================================================================

#include <string.h>

#include "chrx.h"
#include "util.h"

//! How HDF5 tables get created and map to the disc_frame_t structure.
//! See the definitions of #hdf5_table_map_t and #HTAB_ENTRY.
struct hdf5_table_map_t hdf5_table_map[] = {
  HTAB_ENTRY(timestamp),
  HTAB_ENTRY(vis),
  HTAB_ENTRY(vis_flag),
  HTAB_ENTRY(fpga_hk),
  HTAB_ENTRY(serial_adc),
  HTAB_ENTRY_END
};



//------------------------------------------------------------------------------
//! A thread-safe way to access the current frame page being filled.
//!
//! \param self The main acquisition object.
//!
//! \return The index of self->frame that is currently being filled.
//------------------------------------------------------------------------------

int get_frame_page(struct chrx_acq_t *self) {
  int ret;
  
  pthread_mutex_lock(&self->frame_lock);
  ret = (self->frame_page);
  pthread_mutex_unlock(&self->frame_lock);

  return ret;
}


//------------------------------------------------------------------------------
//! Check HDF5 string length, and truncate if needed.
//!
//! \param str The string to check.
//!
//! \return The string, truncated as necessary. The return value should be
//!         treated as read-only.
//------------------------------------------------------------------------------

char *check_hdf5_string(struct chrx_acq_t *self, const char *str) {
  static char ret[HDF5_NAME_LEN];

  if (strlen(str) >= HDF5_NAME_LEN)
    //mlog(M_WARN, "Warning: HDF5 string \"%s\" exceeds %d characters. "
    //             "Truncating.\n", str, HDF5_NAME_LEN - 1);
  strncpy(ret, str, HDF5_NAME_LEN);
  ret[HDF5_NAME_LEN - 1] = '\0';

  return ret;
}


//------------------------------------------------------------------------------
//! Create the serial ADC to frame map.
//!
//! \param self The main acquisition object.
//------------------------------------------------------------------------------

int map_ser_adc_to_frame(struct chrx_acq_t *self) {
/*
    int i, n, curr_index, bad, cal_found;
  char *curr_cal, *curr_name;
  PyObject *dict, *cal_dict, *key, *val;
  Py_ssize_t j, k;

  // Get the channel and calibration dictionaries from the configuration 
  // dictionary.
  dict = grab_conf_dict(self, "acq.serial.channel");
  cal_dict = grab_conf_dict(self, "acq.cal");

  // Allocate.
  n = PyDict_Size(dict);
  self->ser_adc_frame_map = (struct ser_adc_frame_map_t *)
                            malloc(n * sizeof(struct ser_adc_frame_map_t));

  for (i = 0, j = 0; PyDict_Next(dict, &j, &key, &val); i++) {
    curr_name = PyString_AsString(key);

    bad = 0;
    if (!PyList_Check(val))
      bad = 1;
    else if (PyList_Size(val) != 2)
      bad = 1;
    else {
      if (!PyString_Check(PyList_GetItem(val, 0)))
        bad = 1;
      else
        curr_index = atoi(PyString_AsString(PyList_GetItem(val, 0)));
      if (!PyString_Check(PyList_GetItem(val, 1)))
        bad = 1;
      else
        curr_cal = PyString_AsString(PyList_GetItem(val, 1));
    }

    if (bad) {
      //mlog(M_ERR, "Error reading acq.serial.channel.%s in configuration!\n",
      //            curr_name);
      n--;
      self->ser_adc_frame_map = (struct ser_adc_frame_map_t *)
            realloc(self->ser_adc_frame_map, 
                     n * sizeof(struct ser_adc_frame_map_t));
      i--;
      continue;
    }

    // Make sure there is a calibration entry in the configuration
    // corresponding to the value passed.
    cal_found = 0;
    for (k = 0; PyDict_Next(cal_dict, &k, &key, &val);) {
      if (!strcmp(PyString_AsString(key), curr_cal)) {
        cal_found = 1;
        break;
      }
    }

    if (!cal_found) {
      //mlog(M_CRIT, "Calibration \"%s\" for serial ADC channel \"%s\" has no "
      //            "corresponding entry in acq.cal.\n", curr_cal, curr_name);
      exit(0);
    }

    // Populate the map.
    strcpy(self->ser_adc_frame_map[i].name, check_hdf5_string(self, curr_name));
    self->ser_adc_frame_map[i].adc = curr_index;
    strcpy(self->ser_adc_frame_map[i].cal, check_hdf5_string(self, curr_cal));
  }

  return n;
  */
return 0;
}
