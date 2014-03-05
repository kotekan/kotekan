//! \file receiver.c
//! The module for receiving packets from the FPGA and writing HDF5 files.
//==============================================================================

#include <stdio.h>
#include <pthread.h>

#include "chrx.h"

/*
pthread_t rec_loop;
int thread_kill = 0;

void *rec_thread(void *arg) {
  int i, port;

  port = *((int *)arg);

  printf("Starting thread! Port = %d.\n", port);

  for (i = 0;; i++) {
    printf("--> %d\n", i);

    if (thread_kill) {
      printf("Stopping thread!\n");
      thread_kill = 0;

      return NULL;
    }

    usleep(1000000);
  }
}


//==============================================================================
// MODULE METHODS
//==============================================================================

static PyObject *chrx_acquire(struct chrx_t *self, PyObject *args,
                               PyObject *keys) {
  char *keywords[] = {"port", NULL};
  int port;

  if (!PyArg_ParseTupleAndKeywords(args, keys, "i", keywords, &port))
    return NULL;

  pthread_create(&rec_loop, NULL, rec_thread, &port);

  return PyInt_FromLong(0);
}

static PyObject *chrx_stop(struct chrx_t *self, PyObject *args,
                            PyObject *keys) {
  thread_kill = 1;

  return PyInt_FromLong(0);
}
*/

//==============================================================================
// MODULE SETUP
//==============================================================================
/*
static PyMethodDef chrx_methods[] = {
  {"acquire", (PyCFunction)chrx_acquire,
   METH_VARARGS | METH_KEYWORDS, "Testing"},
  {"stop", (PyCFunction)chrx_stop,
   METH_VARARGS | METH_KEYWORDS, "Testing"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initchrx(void) {
  PyObject *mod;

  chrx_acq.tp_new = PyType_GenericNew;
  if (PyType_Ready(&chrx_acq) < 0)
    return;

  // Methods.
  mod = Py_InitModule3("chrx", chrx_methods, "Receiver module.");
  if (mod == NULL)
    return;

  Py_INCREF(&chrx_acq);
  PyModule_AddObject(mod, "acq", (PyObject *)&chrx_acq);

  return;
}
*/

// Hacked a little bit ;)
int chrx_acq_init(struct chrx_acq_t *self) {
  char *keywords[] = {"conf", "log", NULL};
  int i, n;

  // Initialise and set internal values that cannot be configured.
  self->head = NULL;
  self->path_prefix = NULL;
  self->running = 0;
  self->fpga_loop = 0;
  self->ser_adc_loop = 0;
  self->disc_loop = 0;
  pthread_mutex_init(&self->frame_lock, NULL);
  pthread_mutex_init(&self->frame_lock2, NULL);

  // Grab configuration values and derived values.
  self->n_freq = 1024;
  self->n_ant = 16;
  self->n_corr = self->n_ant * (self->n_ant + 1) / 2;
  self->udp_max_len = 0;  // not used.
  self->udp_n_buf = 0; // not used
  self->n_frame_page = 5;
  self->ser_dev = 0;
  self->ser_timeout = 0;
  self->udp_port = 0;
  self->udp_spf = 0;
  self->frame_per_file = 360;

  // Process the serial ADC channel configuration.
  self->n_ser_adc = 0;

  // To be obsoleted.
  self->n_adc = 1;

  // Allocations.
  n = self->n_freq * self->n_corr;
  self->frame = (struct disc_frame_t *)malloc(self->n_frame_page *
                                               sizeof(struct disc_frame_t));
  for (i = 0; i < self->n_frame_page; i++) {
    self->frame[i].timestamp = (struct timestamp_t *)
                                malloc(sizeof(struct timestamp_t));
    self->frame[i].vis = (complex_int_t *)malloc(n * sizeof(complex_int_t));
    self->frame[i].vis_flag = (struct vis_flag_t *)malloc(n *
                                                    sizeof(struct vis_flag_t));
    self->frame[i].fpga_hk = (float *)malloc(self->n_adc *
                                                    sizeof(float));
    self->frame[i].serial_adc = (int16_t *)malloc(self->n_ser_adc *
                                                   sizeof(int16_t));
  }
  self->vis_sum = (double complex *)malloc(n * sizeof(double complex));

  self->udp_buf_len = (int *)malloc(self->udp_n_buf * sizeof(int));
  self->udp_buf = (uint8_t **)malloc(self->udp_n_buf * sizeof(uint8_t *));
  for (n = 0; n < self->udp_n_buf; n++)
    self->udp_buf[n] = (uint8_t *)malloc(self->udp_max_len * sizeof(uint8_t));

  return 0;
}

void chrx_acq_delete(struct chrx_acq_t *self) {
  int i, j;

  free(self->path_prefix);
  free(self->ser_dev);
  for (i = 0; i < self->n_frame_page; i++) {
    free(self->frame[i].timestamp);
    free(self->frame[i].vis);
    free(self->frame[i].vis_flag);
    free(self->frame[i].fpga_hk);
    free(self->frame[i].serial_adc);
  }
  free(self->frame);
  free(self->vis_sum);
  for (i = 0; i < self->udp_n_buf; i++)
    free(self->udp_buf[i]);
  free(self->udp_buf);
  for (i = 0; i < self->n_head; i++) {
    free(self->head[i].i.name);
    if (self->head[i].i.type == 's') {
      for (j = 0; j < self->head[i].s.n_val; j++)
        free(self->head[i].s.val[j]);
    }
    free(self->head[i].i.val);
  }
  free(self->head);
  pthread_mutex_destroy(&self->frame_lock);
  pthread_mutex_destroy(&self->frame_lock2);

  return;
}
