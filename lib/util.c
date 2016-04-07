#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

//------------------------------------------------------------------------------
//! Get the difference in fractional seconds between two times.
//!
//! \param tv1 The first timestamp.
//! \param tv2 The second timestamp.
//!
//! \return The fractional number of seconds between the two times.
//------------------------------------------------------------------------------

double tv_difference(const struct timeval *tv1, const struct timeval *tv2) {
  return (double)(tv2->tv_sec  - tv1->tv_sec) +
         (double)(tv2->tv_usec - tv1->tv_usec) * 1e-6;
}

//------------------------------------------------------------------------------
//! Clone a string.
//!
//! \param s The string to be cloned.
//!
//! \return A cloned string.
//------------------------------------------------------------------------------

char *strclone(const char *s) {
  size_t len = strlen(s);
  char *t = (char *)malloc(len + 1);

  strcpy(t, s);

  return t;
}


int64_t mod(int64_t a, int64_t b)
{
    int ret = a % b;
    if(ret < 0)
        ret+=b;
    return ret;
}