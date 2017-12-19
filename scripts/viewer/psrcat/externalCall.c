#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "psrcat.h"


pulsar psr[MAX_PSR]; 

/* Allows psrcat to be called from a program */
/* Returns 0 = okay                          */
/*         1 = file not opened correctly     */
/*         2 = pulsar not found              */
/*         3 = parameter not found           */

int callPsrcat_val(char *catname,char *i_psrname,char *i_param,double *val,double *err,char *ref)
{
  static linkedList list[MAX_PSR];
  static paramtype defParam[MAX_PARAM];

  char sptr[MAX_PLIST][MAX_STRLEN];
  char param[MAX_STRLEN],psrname[MAX_STRLEN];
  char alias1[MAX_ALIASES][MAX_STRLEN],alias2[MAX_ALIASES][MAX_STRLEN];
  int aliasCount=0;
  static int loadit=0;
  int npsr=0;
  int haveName[MAX_PSR];
  char name[MAX_PSR][MAX_NAMELEN];
  int ptr[MAX_PLIST],ierr;

  strcpy(param,i_param);
  strcpy(psrname,i_psrname);

  /* Convert parameter to upper case */
  upperCase(param);
  upperCase(psrname);

  if (loadit==0)
    {
      loadit=1;
      defineParameters(defParam);
    }
  strcpy(name[0],psrname);
  ptr[0] = getParam(param,NULL);  /* Required parameter */
  strcpy(sptr[0],param);
  if (ptr[0]==-1) /* Parameter not known */
    return 3;

  if (strcmp(catname,"public")==0)
    catname = getenv("PSRCAT_FILE");
  if (readCatalogue(psr,list,&npsr,name,1,haveName,"",ptr,sptr,1,catname,0,0,"",defParam,0,alias1,alias2,aliasCount,0)==1)
    return 1;
  if (npsr==0) return 2; /* Pulsar not found */

  /* Must now get required information */
  *val = psr[0].param[ptr[0]].shortVal;
  sscanf(psr[0].param[ptr[0]].err,"%d",&ierr);
  *err = psr[0].param[ptr[0]].error_expand*ierr;
  strcpy(ref,psr[0].param[ptr[0]].ref);
  return 0;
}


int callPsrcat_string(char *catname,char *i_psrname,char *i_param,char *str)
{
  static paramtype defParam[MAX_PARAM];
  static linkedList list[MAX_PSR];
  char sptr[MAX_PLIST][MAX_STRLEN];
  char psrname[MAX_STRLEN],param[MAX_STRLEN];
  char alias1[MAX_ALIASES][MAX_STRLEN],alias2[MAX_ALIASES][MAX_STRLEN];
  int aliasCount=0;
  static int loadit=0;
  int npsr=0;
  int haveName[MAX_PSR];
  char name[MAX_PSR][MAX_NAMELEN];
  int ptr[MAX_PLIST],ierr;

  strcpy(param,i_param);
  strcpy(psrname,i_psrname);

  /* Convert parameter to upper case */
  upperCase(param);
  upperCase(psrname);

  if (loadit==0)
    {
      loadit=1;
      defineParameters(defParam);
    }
  strcpy(name[0],psrname);
  ptr[0] = getParam(param,NULL);  /* Required parameter */
  strcpy(sptr[0],param);
  if (ptr[0]==-1) /* Parameter not known */
    return 3;

  if (strcmp(catname,"public")==0)
    catname = getenv("PSRCAT_FILE");
  if (readCatalogue(psr,list,&npsr,name,1,haveName,"",ptr,sptr,1,catname,0,0,"",defParam,0,alias1,alias2,aliasCount,0)==1)
    return 1;
  if (npsr==0) return 2; /* Pulsar not found */

  /* Must now get required information */
  strcpy(str,psr[0].param[ptr[0]].val);
  return 0;
}
