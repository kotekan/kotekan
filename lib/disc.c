//! \file disc.c
//! Functions for writing acquired data frames to disc.
//==============================================================================

#include "chrx.h"
#include "util.h"
#include "errors.h"

#include <hdf5.h>
#include <hdf5_hl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

//! For keeping track of the HDF5 architecture.
//! The fractional number is incremented if a new dataset or attribute is added
//! (and therefore the HDF5 is backwards compatible). The integer number changes
//! if a dataset/attribute is removed or renamed, or if the HDF5 architecture 
//! is changed (e.g., by adding a group). The integer number tracks major 
//! structural changes, including but not limited to adding a group or renaming 
//! a channel.
#define HDF5_VERSION    2.0

//! The name of the HDF5 group containing calibration information.
#define HDF5_CAL_GROUP  "cal"

//! Required fields in calibration fields.
char *required_cal_field[] = {"formula", "units", NULL};

extern struct hdf5_table_map_t hdf5_table_map[];

//------------------------------------------------------------------------------
//! For creating the timestamp data table.
//!
//! \param arg Unused.
//! \param name The name of the table.
//! \param fd An open HDF5 file.
//!
//! \return The ID of the new table.
//------------------------------------------------------------------------------

hid_t new_hdf5_table_timestamp(struct chrx_acq_t *self, const char *name, hid_t fd) {
  hid_t dt, ret;
  herr_t err;

  // Create timestamp datatype.
  dt = H5Tcreate(H5T_COMPOUND, sizeof(struct timestamp_t));
  err = H5Tinsert(dt, "fpga_count", HOFFSET(struct timestamp_t, fpga_count),
                  H5T_NATIVE_UINT);
  err = H5Tinsert(dt, "cpu_s", HOFFSET(struct timestamp_t, cpu_s),
                  H5T_NATIVE_UINT);
  err = H5Tinsert(dt, "cpu_us", HOFFSET(struct timestamp_t, cpu_us), 
                  H5T_NATIVE_UINT);
  
  // Create the timestamp table and delete the datatype.
  ret = H5PTcreate_fl(fd, name, dt, (hsize_t)1, -1);
  err = H5Tclose(dt);
  
  if (err < 0) {
    ERROR("Could not create HDF5 table %s (err = %d).", name, err);
    exit(0);
  }

  return ret;
}


//------------------------------------------------------------------------------
//! For creating the visibility data table.
//!
//! \param self For passing the main acquisition object.
//! \param name The name of the table.
//! \param fd An open HDF5 file.
//!
//! \return The ID of the new table.
//------------------------------------------------------------------------------

hid_t new_hdf5_table_vis(struct chrx_acq_t *self, const char *name, hid_t fd) {
  hsize_t dim[2];
  hid_t dtc, dt, ret;
  herr_t err;

  // Create the complex datatype.
  dtc = H5Tcreate(H5T_COMPOUND, sizeof(complex_int_t));
  err = H5Tinsert(dtc, "real", HOFFSET(complex_int_t, real),
                  H5T_NATIVE_INT32);
  err = H5Tinsert(dtc, "imag", HOFFSET(complex_int_t, imag),
                  H5T_NATIVE_INT32);

  // Create the datatype---it is an array.
  dim[0] = self->n_corr;
  dim[1] = self->n_freq;
  dt = H5Tarray_create2(dtc, 2, dim);
  
  // Create the timestamp table and delete the datatype.
  ret = H5PTcreate_fl(fd, name, dt, (hsize_t)1, -1);
  err = H5Tclose(dt);
  err = H5Tclose(dtc);
  
  if (err < 0) {
    ERROR("Could not create HDF5 table %s (err = %d).", name, err);
    exit(0);
  }

  return ret;
}


//------------------------------------------------------------------------------
//! For creating the visibility flag data table.
//!
//! \param self For passing the main acquisition object.
//! \param name The name of the table.
//! \param fd An open HDF5 file.
//!
//! \return The ID of the new table.
//------------------------------------------------------------------------------

hid_t new_hdf5_table_vis_flag(struct chrx_acq_t *self, const char *name, 
                              hid_t fd) {
  hsize_t dim[2];
  hid_t dtc, dt, ret;
  herr_t err;

  // Create the vis_flag datatype.
  dtc = H5Tcreate(H5T_COMPOUND, sizeof(struct vis_flag_t));
  err = H5Tinsert(dtc, "fpga", HOFFSET(struct vis_flag_t, fpga),
                  H5T_NATIVE_UINT8);
  err = H5Tinsert(dtc, "rfi", HOFFSET(struct vis_flag_t, rfi),
                  H5T_NATIVE_UINT8);

  // Create the datatype---it is an array.
  dim[0] = self->n_corr;
  dim[1] = self->n_freq;
  dt = H5Tarray_create2(dtc, 2, dim);
  
  // Create the timestamp table and delete the datatype.
  ret = H5PTcreate_fl(fd, name, dt, (hsize_t)1, -1);
  err = H5Tclose(dt);
  err = H5Tclose(dtc);
  
  if (err < 0) {
    ERROR("Could not create HDF5 table %s (err = %d).", name, err);
    exit(0);
  }

  return ret;
}


//------------------------------------------------------------------------------
//! For creating the fpga_hk table.
//!
//! \param self The main acquisition object.
//! \param name The name of the table.
//! \param fd An open HDF5 file.
//!
//! \return The ID of the new table.
//------------------------------------------------------------------------------

hid_t new_hdf5_table_fpga_hk(struct chrx_acq_t *self, const char *name, 
                             hid_t fd) {
  hsize_t dim;
  hid_t dt, ret;
  herr_t err;

  // Create the datatype---it is an array.
  dim = self->n_adc;
  dt = H5Tarray_create2(H5T_NATIVE_FLOAT, 1, &dim);
  
  // Create the table and delete the datatype.
  ret = H5PTcreate_fl(fd, name, dt, (hsize_t)1, -1);
  err = H5Tclose(dt);
  
  if (err < 0) {
    ERROR("Could not create HDF5 table %s (err = %d).", name, err);
    exit(0);
  }

  return ret;
}

//------------------------------------------------------------------------------
//! For creating the serial_adc table.
//!
//! \param self For passing the main acquisition object.
//! \param name The name of the table.
//! \param fd An open HDF5 file.
//!
//! \return The ID of the new table.
//------------------------------------------------------------------------------

hid_t new_hdf5_table_serial_adc(struct chrx_acq_t *self, const char *name,
                                hid_t fd) {
    
    /*
  int i, j, strpos, cal_strlen;
  char *all_cal;
  hsize_t dim;
  hid_t dt, dta, ha, hsa, dt_str, ret;
  herr_t err;

  // Get the total length of the calibration strings and make a long string
  // containing them all.
  for (i = 0, cal_strlen = 0; i < self->n_ser_adc; i++)
    cal_strlen += strlen(self->ser_adc_frame_map[i].cal) + 1;
  all_cal = (char *)malloc(cal_strlen * sizeof(char));
  for (i = 0, j = 0; i < self->n_ser_adc; i++) {
    strcpy((char *)(all_cal + j), self->ser_adc_frame_map[i].cal);
    j += strlen(self->ser_adc_frame_map[i].cal) + 1;
  }

  // Create the datatypes: both for the dataset (dt) and the attribute listing
  // the calibrations.
  dt = H5Tcreate(H5T_COMPOUND, self->n_ser_adc * sizeof(int16_t));
  dta = H5Tcreate(H5T_COMPOUND, cal_strlen * sizeof(char));
  for (i = 0, strpos = 0; i < self->n_ser_adc; i++) {
    // Add the channel for the dataset.
    err = H5Tinsert(dt, self->ser_adc_frame_map[i].name, i * sizeof(int16_t),
                    H5T_NATIVE_INT16);

    // Add the entry for the calibration attribute.
    j = strlen(self->ser_adc_frame_map[i].cal) + 1;
    dt_str = H5Tcopy(H5T_C_S1);
    err = H5Tset_size(dt_str, j);
    err = H5Tinsert(dta, self->ser_adc_frame_map[i].name, strpos, dt_str);
    err = H5Tclose(dt_str);
    strpos += j * sizeof(char);
  }

  // Create the table and delete the datatype.
  ret = H5PTcreate_fl(fd, name, dt, (hsize_t)1, -1);
  
  // Write the calibrations to an attribute attached to the dataset.
  dim = 1;
  hsa = H5Screate_simple(1, &dim, NULL);
  ha = H5Acreate_by_name(fd, name, HDF5_CAL_GROUP, dta, hsa, H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);
  err = H5Awrite(ha, dta, all_cal);
  
  // Garbage collection.
  err = H5Tclose(dt);
  err = H5Tclose(dta);
  err = H5Sclose(hsa);
  err = H5Aclose(ha);
  free(all_cal);

  if (err < 0) {
    ERROR("Could not create HDF5 table %s (err = %d).", name, err);
    exit(0);
  }

  return ret;
  */
    return 0;
}

int open_hdf5(struct chrx_acq_t *self, int n_file) {
  char path[256], *tmpbuf;
  int i, j, k, len; // req_cal_found;
  double x;
  hsize_t dim[2];
  hid_t hdt_tmp, hs_tmp, ha_tmp; //hg_tmp;
  herr_t err;
  struct vis_layout_t vl;
  // PyObject *dict, *cal_dict, *i_key, *j_key, *i_val, *j_val;
  // Py_ssize_t i_pos, j_pos;

  // Open the HDF5 file.
  sprintf(path, "%s/test.%04d.hdf5", self->path_prefix, n_file);
  self->h_fd = H5Fcreate(path, H5F_ACC_TRUNC,H5P_DEFAULT, H5P_DEFAULT);

  // Create the datasets.
  for (i = 0; strlen(hdf5_table_map[i].name); i++)
    hdf5_table_map[i].tab =
        hdf5_table_map[i].init_func(self, hdf5_table_map[i].name, self->h_fd);

  // Create the calibration group.
  self->hg_cal = H5Gcreate2(self->h_fd, HDF5_CAL_GROUP, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);

  // Write the header information currently possessed by self.
  for (i = 0; i < self->n_head; i++) {
    switch (self->head[i].i.type) {
      case 'i':
        H5LTset_attribute_int(self->h_fd, "/", self->head[i].i.name,
                              self->head[i].i.val, self->head[i].i.n_val);
        break;
      case 'd':
        H5LTset_attribute_double(self->h_fd, "/", self->head[i].d.name,
                                 self->head[i].d.val, self->head[i].d.n_val);
        break;
      case 's': default:
        if (self->head[i].s.n_val == 1) {
          H5LTset_attribute_string(self->h_fd, "/", self->head[i].s.name,
                                   self->head[i].s.val[0]);
        }
        else {
          // Create the dataspace and datatype for this array of strings.
          dim[0] = self->head[i].s.n_val;
          hs_tmp = H5Screate_simple(1, dim, NULL);
          hdt_tmp = H5Tcopy(H5T_C_S1);

          // Find the longest string.
          for (j = 0, len = 0; j < self->head[i].s.n_val; j++) {
            if (strlen(self->head[i].s.val[j]) > len)
              len = strlen(self->head[i].s.val[j]);
          }
          err = H5Tset_size(hdt_tmp, len + 1);

          // Create a long string containing all the values, for writing.
          tmpbuf = (char *)malloc((len + 1) * self->head[i].s.n_val *
                                  sizeof(char));
          for (j = 0; j < self->head[i].s.n_val; j++)
            strcpy((char *)(tmpbuf + j * (len + 1)), self->head[i].s.val[j]);

          // Write the values to the attribute.
          ha_tmp = H5Acreate2(self->h_fd, self->head[i].s.name, hdt_tmp,
                              hs_tmp, H5P_DEFAULT, H5P_DEFAULT);
          err = H5Awrite(ha_tmp, hdt_tmp, tmpbuf);

          // Garbage collection.
          err = H5Sclose(hs_tmp);
          err = H5Tclose(hdt_tmp);
          err = H5Aclose(ha_tmp);
          free(tmpbuf);
        }
        break;
    }
  }

  // Write the visibility layout to an attribute.
  dim[0] = self->n_corr;
  hs_tmp = H5Screate_simple(1, dim, NULL);
  hdt_tmp = H5Tcreate(H5T_COMPOUND, sizeof(struct vis_layout_t));
  err = H5Tinsert(hdt_tmp, "ant_chan_a", 
                  HOFFSET(struct vis_layout_t, ant_chan_a), H5T_NATIVE_UINT16);
  err = H5Tinsert(hdt_tmp, "ant_chan_b", 
                  HOFFSET(struct vis_layout_t, ant_chan_b), H5T_NATIVE_UINT16);

  // Write the visibility layout to a contiguous buffer.
  tmpbuf = (char *)malloc(self->n_corr * sizeof(struct vis_layout_t));
  for (i = 0, k = 0; i < self->n_ant; i++) {
    for (j = i; j < self->n_ant; j++, k += sizeof(struct vis_layout_t)) {
      vl.ant_chan_a = i;
      vl.ant_chan_b = j;
      memcpy((void *)(tmpbuf + k), &vl, sizeof(struct vis_layout_t));
    }
  }

  // Write the layout to an attribute.
  ha_tmp = H5Acreate2(self->h_fd, "chan_indices", hdt_tmp, hs_tmp, H5P_DEFAULT,
                      H5P_DEFAULT);
  err = H5Awrite(ha_tmp, hdt_tmp, tmpbuf);

  // Write the HDF5 version number to an attribute.
  x = HDF5_VERSION;
  H5LTset_attribute_double(self->h_fd, "/", "hdf5_version", &x, 1);

  // Write the calibrations to attributes in a group called "cal".
  /*
  cal_dict = grab_conf_dict(self, "acq.cal");
  for (i_pos = 0; PyDict_Next(cal_dict, &i_pos, &i_key, &i_val);) {
    snprintf(tmpstr, 511, "acq.cal.%s", PyString_AsString(i_key));
    dict = grab_conf_dict(self, tmpstr);

    // Each calibration is its own group with in "cal", with attributes attached
    // to it. Create the group.
    hg_tmp = H5Gcreate2(self->hg_cal, PyString_AsString(i_key), H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);

    // Check for the required calibration fields.
    for (i = 0; required_cal_field[i] != NULL; i++) {
      req_cal_found = 0;
      for (j_pos = 0; PyDict_Next(dict, &j_pos, &j_key, &j_val);) {
        if (!strcmp(required_cal_field[i], PyString_AsString(j_key))) {
          req_cal_found = 1;
          break;
        }
      }
      if (!req_cal_found) {
        ERROR("Calibration %s is missing required field \"%s\"!",
                     tmpstr, required_cal_field[i]);
        exit(0);
      }
    }

    // Write the individual attributes to the group.
    snprintf(tmpstr, 511, "/%s/%s", HDF5_CAL_GROUP, PyString_AsString(i_key));
    for (j_pos = 0; PyDict_Next(dict, &j_pos, &j_key, &j_val);) {
      if (!PyString_Check(j_val)) {
        ERROR("Key %s of calibration %s is not a string!",
                    PyString_AsString(j_key), tmpstr);
        continue;
      }
      err = H5LTset_attribute_string(self->h_fd, tmpstr, 
                                     PyString_AsString(j_key),
                                     PyString_AsString(j_val));
    }

    err = H5Gclose(hg_tmp);
  }
  */

  // Garbage collection.
  free(tmpbuf);
  //err = H5Sclose(hs_tmp);
  //err = H5Aclose(ha_tmp);
  err = H5Tclose(hdt_tmp);
  err = H5Fflush(self->h_fd, H5F_SCOPE_GLOBAL);

  INFO("Created correlator output file %s.", path);

  // To keep the compiler happy.
  if (err)
    err = 0;

  return 0;
}

void close_hdf5(struct chrx_acq_t *self) {
  int i;
  herr_t err;

  for (i = 0; strlen(hdf5_table_map[i].name); i++)
    err = H5PTclose(hdf5_table_map[i].tab);
  H5Gclose(self->hg_cal);
  H5Fclose(self->h_fd);

  // To keep the compiler happy.
  if (err)
    err = 0;

  return;
}

void write_hdf5_row(struct chrx_acq_t *self, int page) {
  int i;
  herr_t err;

  // The "pos" element of hdf_table_map_t tells us where in the struct the data
  // begins for each data table. So: cast the frame as a pointer and add the 
  // offset. This gives a pointer to the pointer to the beginning of the data.
  for (i = 0; strlen(hdf5_table_map[i].name); i++)
    err = H5PTappend(hdf5_table_map[i].tab, (hsize_t)1, 
                     *(void **)((void *)&self->frame[page] +
                                hdf5_table_map[i].pos));
  err = H5Fflush(self->h_fd, H5F_SCOPE_GLOBAL);

  if (err)
    ERROR("Error appending data to HDF file!");
}

void *disc_thread(void *arg) {
  int n_file, n_frame, page;
  struct chrx_acq_t *self;

  self = (struct chrx_acq_t *)arg;
  page = 0; //self->n_frame_page;
  INFO("Starting disc-writing thread.");

  for (n_file = 0;; n_file++) {
    // Create the HDF5 file.
    if (open_hdf5(self, n_file))
      ERROR("Problem opening HDF5 file.");

    for (n_frame = 0; n_frame < self->frame_per_file; n_frame++) {
      // Wait for a new disc frame to appear in buffer.
      while (page == get_frame_page(self)) {
        // Make sure we should still be running.
        if (!self->running)
          return NULL;

        usleep(1);
      }

      write_hdf5_row(self, page);
      INFO("Pushed frame page %d to disc. File chunk %d now has %d "
                   "frame%s written.", page, n_file, n_frame + 1, 
                   n_frame ? "s" : "");
      fflush(stdout);

      if (++page >= self->n_frame_page)
        page = 0;
    }

    close_hdf5(self);
  }

  return NULL;
}
