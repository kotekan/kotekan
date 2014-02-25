//! \file serial_adc.c
//! Functions for reading serial ADC housekeeping data.
//==============================================================================

#include "chrx.h"
#include "util.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <termios.h>

void open_adc_serial(struct chrx_acq_t *self) {
  struct termios tio;

  mlog(M_INFO, "Opening %s for reading ADC data.", self->ser_dev);

  if ((self->fd_ser = open(self->ser_dev, O_RDWR | O_NOCTTY)) < 0)
     mlog(M_ERR, "Couldn't open %s for communication with ADC's.", 
                 self->ser_dev);

  // Set up the TTY parameters.  We leave VTIME and VLEN = 0 so that the 
  // read is non-blocking.
  memset((char *)&tio, 0, sizeof(struct termios));
  tio.c_cflag = B57600 | CSIZE | CREAD | CLOCAL;
  tio.c_iflag = IGNPAR | ICRNL;
  tio.c_iflag &= ~(IXON | IXOFF | IXANY);
  tio.c_oflag = 0;
  tio.c_lflag = 0;
  tio.c_cc[VEOF] = 4;

  tcflush(self->fd_ser, TCIFLUSH);
  tcsetattr(self->fd_ser, TCSANOW, &tio);

  return;
}

void *ser_adc_thread(void *arg) {
  char serial_cmd[64];
  unsigned char buf[1024];
  int i, j, k, len, page;
  struct chrx_acq_t *self;
  struct timeval tv_old, tv_now;

  self = (struct chrx_acq_t *)arg;

  mlog(M_INFO, "Starting serial ADC housekeeping acquisition.");

  open_adc_serial(self);

  for (page = self->n_frame_page;;) {
    if (!self->running) {
      close(self->fd_ser);
      return NULL;
    }

    // Check to see if we should poll the ADC serial port.
    if (page != get_frame_page(self)) {
      page = get_frame_page(self);
      for (i = 0; i < self->n_ser_adc; i++) {
        // Request data.
        len = sprintf(serial_cmd, "%d", self->ser_adc_frame_map[i].adc);
        for (j = len; j > 0; j -= k) {
          if ((k = write(self->fd_ser, serial_cmd + len - j, j)) <= 0) {
            mlog(M_ERR, "Could not write to ADC serial (errno = %d).",
                        errno);
            break;
          }
        }

        // Listen for data.
        gettimeofday(&tv_old, NULL);
        for (j = 0; j < sizeof(short);) {
          len = read(self->fd_ser, buf + j, 1024);
          if (len < 0) {
            mlog(M_ERR, "Could not read from ADC serial.");
            break;
          }
          j += len;

          // Don't wait forever---timeout if the response is not coming.
          gettimeofday(&tv_now, NULL);
          if (tv_difference(&tv_old, &tv_now) > self->ser_timeout) {
            mlog(M_ERR, "Timed out waiting for response from ADC serial.");
            break;
          }

          usleep(1);
        }

        self->frame[page].serial_adc[i] = buf[0] << 8 | buf[1];
      }
    }
    usleep(1);
  }
}
