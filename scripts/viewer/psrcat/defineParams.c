#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "psrcat.h"

void defineCustom(pulsar *psr,int ip,int type,int exactMatch);
void defineDegrees(pulsar *psr,int ip,int type);
void defineBinary(pulsar *psr,int ip,int type);
void convertEcliptic(double raj,double decj,double *elong,double *elat);
void defineDistance(pulsar *psr,int ip,int type);
void derive_basedP(pulsar *psr,int ip,int type);
void defineGalactic(pulsar *psr,int ip,int type);
void convertGalactic(double raj,double decj,double *gl,double *gb);
void defineEcliptic(pulsar *psr,int ip,int type);
void defineEquatorial(pulsar *psr,int ip,int type);
void definePosEpoch(pulsar *psr,int ip,int type);
void defineDMEpoch(pulsar *psr,int ip,int type);
void derive_oSurvey(pulsar *psr,int ip,int type);
double galpa(double el, double eb);
void definePM(pulsar *psr,int ip,int type);
void defineFlux(pulsar *psr,int ip,int type);
int encode_survey( char * inptr,int survey_num,char surveynames[MAX_STRLEN][MAX_SURVEYS]);
int define_surveys(char surveynames[MAX_STRLEN][MAX_SURVEYS]);
void addParameter(paramtype *defParam,char *p1,char *p2,char *p3,char *p4,char *p5,int helpType,int ephem);
int nearInt(double x);
double m2(double mf, double sini, double m1);
float posAngle(float a1,float b1,float a2,float b2);

int getParam(char *str,pulsar *psr)
{
  register unsigned int i;
  for (i=0;i<MAX_PARAM;i++)
    {
      if (strcmp(pcat_parameters[i],str)==0)
	{
	  /* Check if we must derive this parameter */
	  if (psr!=NULL && psr->param[i].set1 != 1)
	    deriveParameter(str,psr,i);
	  return i;
	}
    }  
  return -1;
}

void deriveParameter(char *str,pulsar *psr,int ip)
{
  double v1,v2,v3;
  int p0=-1,p1=-1;
  int setP1;
  int i1,i2;

  if (ip<0) return; /* Don't try anything if the parameter does not exist */

  /* **** */
  /* NAME */
  /* **** */
  if (strcmp(str,"NAME")==0 && psr->param[ip].set1 == 0)
    {
      i1 = getParam("PSRB",NULL);
      if (psr->param[i1].set1 == 0) /* Do not have PSRB */
	i1 = getParam("PSRJ",NULL);

      strcpy(psr->param[ip].val,psr->param[i1].val);
      strcpy(psr->param[ip].ref,psr->param[i1].ref);
      psr->param[ip].set1 = 1;
      psr->param[ip].set3 = 1;
      pcat_refZero[ip] = 1;
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);
      return;
    }


  /* ****************************** */
  /* BASED ON PERIOD AND DERIVATIVE */
  /* ****************************** */
  if (strcmp(str,"AGE")==0)   derive_basedP(psr,ip,1);
  if (strcmp(str,"BSURF")==0) derive_basedP(psr,ip,2);
  if (strcmp(str,"B_LC")==0)  derive_basedP(psr,ip,3);
  if (strcmp(str,"EDOT")==0)  derive_basedP(psr,ip,4);
  if (strcmp(str,"EDOTD2")==0)  derive_basedP(psr,ip,5);

  /* *************** */
  /* Based on survey */
  /* *************** */
  if (strcmp(str,"OSURVEY")==0) derive_oSurvey(psr,ip,1);

  /* *********************** */
  /* Based on proper motions */
  /* *********************** */
  if (strcmp(str,"PMTOT")==0) definePM(psr,ip,1);
  else if (strcmp(str,"VTRANS")==0) definePM(psr,ip,2);
  else if (strcmp(str,"P1_I")==0) definePM(psr,ip,3);
  else if (strcmp(str,"AGE_I")==0) definePM(psr,ip,4);
  else if (strcmp(str,"BSURF_I")==0) definePM(psr,ip,5);
  else if (strcmp(str,"EDOT_I")==0) definePM(psr,ip,6);
  else if (strcmp(str,"PML")==0) definePM(psr,ip,7);
  else if (strcmp(str,"PMB")==0) definePM(psr,ip,8);
  else if (strcmp(str,"PMERR_PA")==0) definePM(psr,ip,9);

  /* ****** */
  /* EPOCHS */
  /* ****** */
  if (strcmp(str,"POSEPOCH")==0)  definePosEpoch(psr,ip,1);
  if (strcmp(str,"DMEPOCH")==0)  defineDMEpoch(psr,ip,1);
  if (strcmp(str,"DATE")==0) definePosEpoch(psr,ip,2);

  /* ************** */
  /* RAJD and DECJD */
  /* ************** */
  if (strcmp(str,"RAJD")==0) defineDegrees(psr,ip,1);
  else if (strcmp(str,"DECJD")==0) defineDegrees(psr,ip,2);

  /* ***************** */
  /* Binary Parameters */
  /* ***************** */
  if (strcmp(str,"MASSFN")==0)       defineBinary(psr,ip,1);
  else if (strcmp(str,"MINMASS")==0) defineBinary(psr,ip,2);
  else if (strcmp(str,"MEDMASS")==0) defineBinary(psr,ip,3);
  else if (strcmp(str,"UPRMASS")==0) defineBinary(psr,ip,4);
  else if (strcmp(str,"OM")==0)      defineBinary(psr,ip,5);
  else if (strcmp(str,"ECC")==0)     defineBinary(psr,ip,6);
  else if (strcmp(str,"MINOMDOT")==0)defineBinary(psr,ip,7);

  /* ******************* */
  /* Distance parameters */
  /* ******************* */
  if (strcmp(str,"DIST")==0)  defineDistance(psr,ip,1);
  if (strcmp(str,"DIST1")==0) defineDistance(psr,ip,2);

  /* *************** */
  /* Flux parameters */
  /* *************** */
  if (strcmp(str,"SI414")==0)   defineFlux(psr,ip,1);
  if (strcmp(str,"R_LUM")==0)   defineFlux(psr,ip,2);
  if (strcmp(str,"R_LUM14")==0) defineFlux(psr,ip,3);

  /* ******************** */
  /* Galactic coordinates */
  /* ******************** */
  if (strcmp(str,"GL")==0 || strcmp(str,"GB")==0) defineGalactic(psr,ip,1);
  else if (strcmp(str,"ZZ")==0) defineGalactic(psr,ip,2);
  else if (strcmp(str,"XX")==0) defineGalactic(psr,ip,3);
  else if (strcmp(str,"YY")==0) defineGalactic(psr,ip,4);
  else if (strcmp(str,"DMSINB")==0) defineGalactic(psr,ip,5);

  /* ******************** */
  /* Ecliptic coordinates */
  /* ******************** */
  if (strcmp(str,"ELONG")==0 || strcmp(str,"ELAT")==0) defineEcliptic(psr,ip,1);
  if (strcmp(str,"RAJ")==0 || strcmp(str,"DECJ")==0 
      || strcmp(str,"RAJD")==0 || strcmp(str,"DECJD")==0) defineEquatorial(psr,ip,1);


  /* ***************** */
  /* Custom parameters */
  /* ***************** */
  if (strcmp(str,"C1")==0) defineCustom(psr,ip,1,0);
  else if (strcmp(str,"C2")==0) defineCustom(psr,ip,2,0);
  else if (strcmp(str,"C3")==0) defineCustom(psr,ip,3,0);
  else if (strcmp(str,"C4")==0) defineCustom(psr,ip,4,0);
}

/* *************** */
/* Based on survey */
/* *************** */
void derive_oSurvey(pulsar *psr,int ip,int type)
{
  int osurvey_val,ip1;
  char old_survey[MAX_STRLEN];
  static char surveynames[MAX_STRLEN][MAX_SURVEYS];
  static int surveyNum=0;

  ip1 = getParam("SURVEY",NULL);
  if (surveyNum==0)
    surveyNum = define_surveys(surveynames);

  if (psr->param[ip1].set1==1)
    {
      strcpy(old_survey,psr->param[ip1].val);
      osurvey_val = encode_survey(old_survey,surveyNum,surveynames);
      psr->param[ip].set1 = 1; psr->param[ip].shortVal = (double)osurvey_val; 
      sprintf(psr->param[ip].val,"%10.10o",osurvey_val);  
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);              
    }
}




/* ************** */
/* Position Epoch */
/* ************** */
void definePosEpoch(pulsar *psr,int ip,int type)
{
  int ip1,len,date,i;
  char ref[4];
  double val1;
  /* If posepoch is not set, then set it to the period epoch */
  if (type==1)
    {
      ip1 = getParam("PEPOCH",NULL);
      if (psr->param[ip1].set1==1)
	{
	  val1 = psr->param[ip1].shortVal;
	  psr->param[ip].set1 = 1; psr->param[ip].shortVal = val1; sprintf(psr->param[ip].val,"%8.2f",val1);  
	  psr->param[ip].derived = 1;
	  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);              
	}
    }
  else if (type==2) /* Discovery date */
    {
      if (strcmp(psr->param[0].ref,NO_REF)!=0) /* Obtain from the reference to the PSRJ name */
	{
	  len = strlen(psr->param[0].ref);
	  /* go from the end and find the first numeral */
	  /* In case the reference is something like abc45a */
	  for (i=len;i--;)
	    {
	      if (isdigit(psr->param[0].ref[i]))
		{
		  len = i;
		  break;
		}
	    }
	  ref[0] = psr->param[0].ref[len-1];
	  ref[1] = psr->param[0].ref[len];
	  ref[2] = '\0';
	  /* Have only the last two digits of the date */
	  sscanf(ref,"%d",&date);
	  if (date > 65)
	    date = 1900+date;
	  else
	    date = 2000+date;
	  psr->param[ip].set1 = 1; psr->param[ip].shortVal = date; sprintf(psr->param[ip].val,"%-5d",date);  
	  psr->param[ip].derived = 1;
	  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);             
	}      
    }
}

/* ************** */
/* DM Epoch */
/* ************** */
void defineDMEpoch(pulsar *psr,int ip,int type)
{
  int ip1,len,date,i;
  char ref[4];
  double val1;
  /* If dmsepoch is not set, then set it to the period epoch */
  ip1 = getParam("PEPOCH",NULL);
  if (psr->param[ip1].set1==1)
    {
      val1 = psr->param[ip1].shortVal;
      psr->param[ip].set1 = 1; psr->param[ip].shortVal = val1; sprintf(psr->param[ip].val,"%8.2f",val1);  
      psr->param[ip].derived = 1;
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);              
    }
}


void defineFlux(pulsar *psr,int ip,int type)
{
  int ip1,ip2,ip3;
  double val1,val2,val3,val;

  ip1 = getParam("S1400",NULL);
  ip2 = getParam("S400",NULL);
  
  if (type==1 && psr->param[ip1].set1==1 && psr->param[ip2].set1==1) /* SI414 */
    {
      val1 = psr->param[ip2].shortVal;
      val2 = psr->param[ip1].shortVal;
      val  = -log10(val1/val2)/(log10(400.0/1400.0));
      psr->param[ip].set1 = 1; psr->param[ip].shortVal = val; sprintf(psr->param[ip].val,"%-6.2f",val);  
      psr->param[ip].derived = 1;
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);            
    }
  else if (type==2 && psr->param[ip2].set1==1) /* R_LUM */
    {
      ip3 = getParam("DIST",psr);
      if (psr->param[ip3].set1==1)
	{
	  val1 = psr->param[ip2].shortVal;
	  val2 = psr->param[ip3].shortVal;
	  val3 = val1*val2*val2;
	  psr->param[ip].set1 = 1; psr->param[ip].shortVal = val3; sprintf(psr->param[ip].val,"%-8.2f",val3); 
	  psr->param[ip].derived = 1;
	  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);            
	}
    }
  else if (type==3 && psr->param[ip1].set1==1) /* R_LUM14 */
    {
      ip3 = getParam("DIST",psr);
      if (psr->param[ip3].set1==1)
	{
	  val1 = psr->param[ip1].shortVal;
	  val2 = psr->param[ip3].shortVal;
	  val3 = val1*val2*val2;
	  psr->param[ip].set1 = 1; psr->param[ip].shortVal = val3; sprintf(psr->param[ip].val,"%-8.2f",val3);  
	  psr->param[ip].derived = 1;
	  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);            
	}
    }
    
}
/* *********************** */
/* Based on proper motions */
/* *********************** */
void definePM(pulsar *psr,int ip,int type)
{
  int ip1,ip2,ip3,ip4,ierr1,ierr2,lv,le,ip5,ip6,ip7,ip8,ecliptic;
  char msg[MAX_STRLEN];
  double val,val1,val2,val3,val4,val5,mul,mub,mulerr,muberr;
  double err,err1,err2,err3;

  if (type==1) /* PMTOT */
    {
      ip1 = getParam("PMRA",psr);
      ip2 = getParam("PMDEC",psr);
      if (psr->param[ip1].set1 != 1 || psr->param[ip2].set1 != 1)
	{
	  ip1 = getParam("PMELONG",psr);
	  ip2 = getParam("PMELAT",psr);	  
	}
      
           
      if (psr->param[ip1].set1 == 1 && psr->param[ip2].set1 == 1)
	{
	  val1 = psr->param[ip1].shortVal;
	  val2 = psr->param[ip2].shortVal;
	
	  sscanf(psr->param[ip1].err,"%d",&ierr1);
	  sscanf(psr->param[ip2].err,"%d",&ierr2);	      
	  err1 = psr->param[ip1].error_expand*ierr1;
	  err2 = psr->param[ip2].error_expand*ierr2;
	  err  = sqrt((pow(val1*err1,2)+pow(val2*err2,2))/(val1*val1 + val2*val2));
	  val  = sqrt(val1*val1+val2*val2);
	  rnd8(val,err,1,psr->param[ip].val,&lv,psr->param[ip].err,&le,msg);
	  psr->param[ip].shortVal = val;
	  psr->param[ip].error_expand = errscale(psr->param[ip].val,psr->param[ip].err);
	  psr->param[ip].derived = 1;
	  psr->param[ip].set1 = 1;
	  psr->param[ip].set2 = 1;
	  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val); 
	  pcat_errZero[ip]=1;
	}
      else
	{
	  strcpy(psr->param[ip].err,"0");
	  psr->param[ip].error_expand = 0;
	  psr->param[ip].derived = 1;
	  psr->param[ip].set2 = 1;
	  pcat_errZero[ip]=1;
	}
    }
  else if (type==2) /* VTRANS */
    {
      ip1 = getParam("PMTOT",psr);
      ip2 = getParam("DIST",psr);
      if (psr->param[ip1].set1 == 1 && psr->param[ip2].set1 == 1)
	{
	  val1 = psr->param[ip1].shortVal;
	  val2 = psr->param[ip2].shortVal;

	  val = val1/1000.0/3600.0/180.0*M_PI/365.25/86400.0*3.086e16*val2;
	  /*	  printf("%s Using %f %f %f\n",psr->param[0].val,val1,val2,val); */
	  psr->param[ip].set1 = 1; psr->param[ip].shortVal = val; sprintf(psr->param[ip].val,"%-6.2f",val);
	  psr->param[ip].derived = 1;
   	  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);
	}
    }
  else if (type==3 || type==4 || type==5 || type==6) /* P1_I/AGE_I/BSURF_I/EDOT_I */
    {
      ip1 = getParam("VTRANS",psr);
      ip2 = getParam("P1",NULL);
      ip3 = getParam("P0",NULL);
      ip4 = getParam("DIST",psr);
      ip5 = getParam("P1_I",NULL);
      ip6 = getParam("AGE_I",NULL);
      ip7 = getParam("BSURF_I",NULL);
      ip8 = getParam("EDOT_I",NULL);
      if (psr->param[ip1].set1 == 1)
	{
	  val1 = psr->param[ip1].shortVal;
	  val2 = psr->param[ip2].shortVal;
	  val3 = psr->param[ip3].shortVal;
	  val4 = psr->param[ip4].shortVal;

	  /* Set P1_I */
	  val = (val2/1.0e-15 - pow(val1,2)*1.0e10*val3/(val4*3.086e6)/2.9979e10)*1.0e-15;
	  sprintf(psr->param[ip5].val,"%.2e",val);
	  psr->param[ip5].shortVal = val;
	  psr->param[ip5].set1 = 1;
	  psr->param[ip5].derived = 1;
	  if (pcat_maxSize[ip5] < strlen(psr->param[ip5].val)) pcat_maxSize[ip5]=strlen(psr->param[ip5].val);

	  /* Set AGE_I */
	  val5 = (val3/2.0/val)/60.0/60.0/24.0/365.25; /* Age in Yr */
	  sprintf(psr->param[ip6].val,"%.2e",val5);
	  psr->param[ip6].shortVal = val5;
	  psr->param[ip6].set1 = 1;
	  psr->param[ip6].derived = 1;
	  if (pcat_maxSize[ip6] < strlen(psr->param[ip6].val)) pcat_maxSize[ip6]=strlen(psr->param[ip6].val);	  

	  /* Set BSURF_I */
	  val5 = 3.2e19*sqrt(val3*val);
	  sprintf(psr->param[ip7].val,"%.2e",val5);
	  psr->param[ip7].shortVal = val5;
	  psr->param[ip7].set1 = 1;
	  psr->param[ip7].derived = 1;
	  if (pcat_maxSize[ip7] < strlen(psr->param[ip7].val)) pcat_maxSize[ip7]=strlen(psr->param[ip7].val);	  

	  /* Set EDOT_I */
	  val5 = 4.0 /1.0e-15 * M_PI * M_PI * val / pow(val3,3)*1.0e30;  
	  sprintf(psr->param[ip8].val,"%.2e",val5);
	  psr->param[ip8].shortVal = val5;
	  psr->param[ip8].set1 = 1;
	  psr->param[ip8].derived = 1;
	  if (pcat_maxSize[ip8] < strlen(psr->param[ip8].val)) pcat_maxSize[ip8]=strlen(psr->param[ip8].val);	  
	}
    }
  else if (type==7 || type==8) /* PML and PMB */
    {
      ip1 = getParam("PMELAT",psr);
      ip2 = getParam("PMELONG",psr);
      if (psr->param[ip1].set1 != 1 || psr->param[ip2].set1 != 1)
	{
	  ip1 = getParam("PMRA",psr);
	  ip2 = getParam("PMDEC",psr);
	  ip3 = getParam("RAJD",psr);
	  ip4 = getParam("DECJD",psr);
	  ecliptic=0;
	}
      else
	{
	  ip3 = getParam("ELONG",psr);
	  ip4 = getParam("ELAT",psr);
	  ecliptic = 1;
	}
      ip5 = getParam("DIST",psr);
      if (psr->param[ip1].set1 == 1 && psr->param[ip2].set1 == 1 && psr->param[ip5].set1 == 1)
	{
	  val1 = psr->param[ip1].shortVal;
	  val2 = psr->param[ip2].shortVal;
	  val3 = psr->param[ip3].shortVal;
	  val4 = psr->param[ip4].shortVal;
	  val5 = psr->param[ip5].shortVal;
	  
	  sscanf(psr->param[ip1].err,"%d",&ierr1);
	  sscanf(psr->param[ip2].err,"%d",&ierr2);	      
	  err1 = psr->param[ip1].error_expand*ierr1;
	  err2 = psr->param[ip2].error_expand*ierr2;

	  /* Have pmra (val1), pmdec (val2), rajd (val3) and decjd (val4) */
	  /* Must calculate pml and pmb using hla93 method                */
	  /* dist = val5                                                  */	 
	  if (ecliptic==0) find_galcoord_equatorial(val3*M_PI/180.0,val4*M_PI/180.0,val1,err1,val2,err2,
					 val5,&mul,&mulerr,&mub,&muberr);
	  else find_galcoord_ecliptic(val3*M_PI/180.0,val4*M_PI/180.0,val1,err1,val2,err2,
				      val5,&mul,&mulerr,&mub,&muberr);
	  ip = getParam("PML",NULL);
	  rnd8(mul,mulerr,1,psr->param[ip].val,&lv,psr->param[ip].err,&le,msg);
	  psr->param[ip].shortVal = mul;
	  psr->param[ip].error_expand = errscale(psr->param[ip].val,psr->param[ip].err);
	  psr->param[ip].set1 = 1;
	  psr->param[ip].set2 = 1;
	  psr->param[ip].derived = 1;
	  pcat_errZero[ip]=-1;

	  ip = getParam("PMB",NULL);
	  rnd8(mub,muberr,1,psr->param[ip].val,&lv,psr->param[ip].err,&le,msg);
	  psr->param[ip].shortVal = mub;
	  psr->param[ip].error_expand = errscale(psr->param[ip].val,psr->param[ip].err);
	  psr->param[ip].set1 = 1;
	  psr->param[ip].set2 = 1;
	  psr->param[ip].derived = 1;
	  pcat_errZero[ip]=-1;
	}
    }
  else if (type==9) /* PMERR_PA */
    {
      double pmerrPA;
      ip1 = getParam("ELONG",psr);
      ip2 = getParam("ELAT",psr);
      pmerrPA = galpa(psr->param[ip1].shortVal,psr->param[ip2].shortVal);
      rnd8(pmerrPA,0.01,1,psr->param[ip].val,&lv,psr->param[ip].err,&le,msg);
      psr->param[ip].shortVal = pmerrPA;
      psr->param[ip].set1 = 1;
      psr->param[ip].set2 = 0;
      psr->param[ip].derived = 1;
      pcat_errZero[ip]=-1;
    }
}

/* ***************** */
/* Define Equatorial */
/* ***************** */
void defineEquatorial(pulsar *psr,int ip,int type)
{
  int ip1,ip2,ip3,ip4,ip5,ip6,ip7,ierr,dp,err,i,sym;
  double val,val1,val2,val3,elong,elat;
  double ce = 0.91748213149438; /* Cos epsilon  */
  double se = 0.39777699580108; /* Sine epsilon */
  double dr = M_PI/180.0;
  double cos_ra,sin_ra,ra,ddb,ddl,dece,g,k,y,z,dzdl,dzdb,dddb,dddl,dal,dab,rae;
  double el,ele,eb,ebe,elb,sdec,dec,dummy;
  char rajC[500],decjC[500],sdummy[100],sd2[100];

  if (type==1) /* RAJ/DECJ */
    {
      ip1 = getParam("RAJD",NULL);
      ip2 = getParam("DECJD",NULL);
      ip3 = getParam("ELONG",NULL);
      ip4 = getParam("ELAT",NULL);
      ip5 = getParam("POSEPOCH",psr);
      ip6 = getParam("RAJ",NULL);
      ip7 = getParam("DECJ",NULL);

      if (psr->param[ip3].set1==1 && psr->param[ip4].set1==1)
	{
	  el = psr->param[ip3].shortVal; /* Elong */
	  eb = psr->param[ip4].shortVal; /* Elat  */
	  sscanf(psr->param[ip3].err,"%d",&ierr);	      
	  ele = psr->param[ip3].error_expand*ierr;
	  sscanf(psr->param[ip4].err,"%d",&ierr);	      
	  ebe = psr->param[ip4].error_expand*ierr;
	  val3 = psr->param[ip5].shortVal;

	  sdec  = sin(eb*dr)*ce + cos(eb*dr)*se*sin(el*dr);
	  dec   = asin(sdec);        /* in radians */
	  cos_ra = (cos(el*dr)*cos(eb*dr)/cos(dec));
	  sin_ra = (sin(dec)*ce - sin(eb*dr))/(cos(dec)*se);
	  ra = atan2(sin_ra,cos_ra); /* in radians */
	  /* Get RA into range 0 to 360 */
	  if (ra < 0.0) ra = 2*M_PI+ra;
	  else if (ra > 2*M_PI) ra = ra - 2*M_PI; 
	  
	  ddb = 1.0/sqrt(1.0-sdec*sdec)*(cos(eb*dr)*ce - sin(eb*dr)*sin(el*dr)*se);
	  ddl = 1.0/sqrt(1.0-sdec*sdec)*cos(eb*dr)*cos(el*dr)*se;
	  dece = sqrt(ddb*ddb*ebe*ebe + ddl*ddl*ele*ele);
	  g = sin(eb*dr)*ce+cos(eb*dr)*se*sin(el*dr);
	  k = asin(g);
	  y = cos(k);
	  z = 1.0/y;
	  
	  dzdl = sin(k)/y/y*cos(eb*dr)*se*cos(el*dr)/sqrt(1.0-g*g);
	  dzdb = sin(k)/y/y*1.0/sqrt(1.0-g*g)*(cos(eb*dr)*ce-sin(eb*dr)*se*sin(el*dr));
	  dal = -1.0/sqrt(1.0-cos(ra)*cos(ra))*(-cos(eb*dr)*sin(el*dr)*z +
                                       cos(el*dr)*cos(eb*dr)*dzdl);

	  dab = -1.0/sqrt(1.0-cos(ra)*cos(ra))*(-cos(el*dr)*sin(eb*dr)*z +
						cos(el*dr)*cos(eb*dr)*dzdb);
	  
	  rae = sqrt(dab*dab*ebe*ebe + dal*dal*ele*ele); 
	  
	  /* Now have raj, decj in radians */
	  
	  /* Set rajd and decjd */	 
	  psr->param[ip1].set1 = 1; psr->param[ip1].shortVal = ra/dr; sprintf(psr->param[ip1].val,"%-10.5f",ra/dr);      
	  psr->param[ip2].set1 = 1; psr->param[ip2].shortVal = dec/dr; sprintf(psr->param[ip2].val,"%-10.5f",dec/dr);        
	  psr->param[ip1].derived = 1;	  psr->param[ip2].derived = 1;
	  if (pcat_maxSize[ip1] < strlen(psr->param[ip1].val)) pcat_maxSize[ip1]=strlen(psr->param[ip1].val);
	  if (pcat_maxSize[ip2] < strlen(psr->param[ip2].val)) pcat_maxSize[ip2]=strlen(psr->param[ip2].val);  

	  /* Now calculate raj and decj */
	  turn_dms(dec/2.0/M_PI,decjC);
	  turn_hms(ra/2.0/M_PI,rajC);
	  /* Now convert this to the correct number of decimal places for printing */

	  /* DECJ */
	  dp = nearInt(log10(dece*3600.0));
	  if (dp <= 0)
	    {
	      if (dece*3600.0/pow(10,dp)<1.90)
		{
		  if (dp <= 0) dp --;
		  else dp ++;
		}
	    }
	  if (dp <= 0)
	    err = (int)(dece*3600.0/pow(10,dp)+0.9999);
	  else
	    err = (int)(dece*3600.0+0.9999);

	  sym=0;


	  for (i=0;i<strlen(decjC);i++)
	    {
	      if (decjC[i] == ':') sym=i;
	      if (decjC[i] == '.' && dp < 0) {
		if (decjC[i-dp+1] >= '5' && decjC[i-dp+1] <= '9')  /* Round up */
		  {
		    sscanf(decjC+sym+1,"%lf",&dummy);
		    dummy/=pow(10,dp);
		    dummy = (double)nint(dummy)*pow(10,dp);
		    decjC[i-dp]=decjC[i-dp]+1;
		    decjC[sym+1]='\0';
		    if (dummy<10)
		      sprintf(sd2,"0%%.%df",-dp);
		    else
		      sprintf(sd2,"%%.%df",-dp);
		    sprintf(sdummy,sd2,dummy);
		    strcat(decjC,sdummy);
		  }
		else
		  {
		    decjC[i-dp+1]='\0'; 
		  }
		break;
	      }
	      else if (decjC[i] == '.' && dp >= 0)
		{
		  if (decjC[i+1] >= '5' && decjC[i+1] <= '9')  /* Round up */
		  {
		    sscanf(decjC+sym+1,"%lf",&dummy);
		    dummy = (double)nint(dummy);
		    decjC[i]=decjC[i]+1;
		    decjC[sym+1]='\0';
		    sprintf(sd2,"%%02d");
		    sprintf(sdummy,sd2,(int)dummy);
		    strcat(decjC,sdummy);
		    break;
		  }
		else
		  {
		    decjC[i] = '\0';
		    break;
		  }
		}
	    }

	  psr->param[ip7].set1 = 1; strcpy(psr->param[ip7].val,decjC);
	  psr->param[ip7].set2 = 1; sprintf(psr->param[ip7].err,"%d",err);
	  if (psr->param[ip3].set3 == 1) {psr->param[ip7].set3 = 1; 
	  strcpy(psr->param[ip7].ref,psr->param[ip3].ref); pcat_refZero[ip7]=-1;}
	  psr->param[ip7].error_expand = dece*3600.0/(double)err;
	  psr->param[ip7].derived = 1;	  
	  pcat_errZero[ip7]=-1;
	  if (pcat_maxSize[ip7] < strlen(psr->param[ip7].val)) pcat_maxSize[ip7]=strlen(psr->param[ip7].val);

	  /* ******************************************* */
	  /* RAJ                                         */
	  /* ******************************************* */

	  dp = nearInt(log10(rae*3600.0/15.0));
	  if (rae*3600.0/15.0/pow(10,dp)<1.90)
	    {
	      if (dp < 0) dp --;
	      else dp ++;
	    }
	  err = (int)(rae*3600.0/15.0/pow(10,dp)+0.9999);
	  sym=0;
	  for (i=0;i<strlen(rajC);i++)
	    {
	      if (rajC[i] == ':') sym++;
	      if (rajC[i] == '.' && dp < 0) {rajC[i-dp+1]='\0'; break;}
	    }
	  psr->param[ip6].set1 = 1; strcpy(psr->param[ip6].val,rajC+1);
	  psr->param[ip6].set2 = 1; sprintf(psr->param[ip6].err,"%d",err);
	  if (psr->param[ip3].set3 == 1) {psr->param[ip6].set3 = 1; 
	  strcpy(psr->param[ip6].ref,psr->param[ip3].ref); pcat_refZero[ip6]=-1;}
	  psr->param[ip6].error_expand = rae*3600.0/15.0/(double)err;
	  psr->param[ip6].derived = 1;	  
	  pcat_errZero[ip6]=-1;
	  if (pcat_maxSize[ip6] < strlen(psr->param[ip6].val)) 
	    pcat_maxSize[ip6]=strlen(psr->param[ip6].val);

	  /*	  printf("Have %s (%f) %d %d %s \n",rajC,rae*3600.0/15.0,dp,err,decjC,dece*3600.0); */
	} 
    } 
}

/* ***************** */
/* Define Ecliptic   */
/* ***************** */
void defineEcliptic(pulsar *psr,int ip,int type)
{
  int ip1,ip2,ip3,ip4,ip5;
  double val,val1,val2,val3,elong,elat;
  double deg2rad = M_PI/180.0;

  if (type==1) /* ELONG/ELAT */    
    {
      ip1 = getParam("RAJD",psr);
      ip2 = getParam("DECJD",psr);
      ip3 = getParam("ELONG",NULL);
      ip4 = getParam("ELAT",NULL);
      ip5 = getParam("POSEPOCH",psr);

      if (psr->param[ip1].set1==1 && psr->param[ip2].set1==1)
	{
	  val1 = psr->param[ip1].shortVal*deg2rad;
	  val2 = psr->param[ip2].shortVal*deg2rad;
	  val3 = psr->param[ip5].shortVal;

	  /*	  sla_eqecl_(&val1,&val2,&val3,&elong,&elat); */ /* Call SLALIB routine to convert to elong,elat */
	  /*	  slaEqecl(val1,val2,val3,&elong,&elat); *********** */
	  convertEcliptic(val1,val2,&elong,&elat);
	  /*	  slaEqecl(val1,val2,val3,&elong,&elat); */
	  elong/=deg2rad;
	  elat/=deg2rad;
	  psr->param[ip3].set1 = 1; psr->param[ip3].shortVal = elong; sprintf(psr->param[ip3].val,"%-6.2f",elong);      
	  psr->param[ip4].set1 = 1; psr->param[ip4].shortVal = elat; sprintf(psr->param[ip4].val,"%-6.2f",elat);        
	  psr->param[ip3].derived = 1;	  psr->param[ip4].derived = 1;
	  if (pcat_maxSize[ip3] < strlen(psr->param[ip3].val)) pcat_maxSize[ip3]=strlen(psr->param[ip3].val);
	  if (pcat_maxSize[ip4] < strlen(psr->param[ip4].val)) pcat_maxSize[ip4]=strlen(psr->param[ip4].val);
	}
    }
}
/* ***************** */
/* Distance Galactic */
/* ***************** */
void defineGalactic(pulsar *psr,int ip,int type)
{
  int ip1,ip2,ip3,ip4;
  double val,val1,val2,val3,gl,gb;
  double deg2rad = M_PI/180.0;

  if (type==1) /* GL/GB */    
    {
      ip1 = getParam("RAJD",psr);
      ip2 = getParam("DECJD",psr);
      ip3 = getParam("GL",NULL);
      ip4 = getParam("GB",NULL);
      
      if (psr->param[ip1].set1==1 && psr->param[ip2].set1==1)
	{
	  val1 = psr->param[ip1].shortVal*deg2rad;
	  val2 = psr->param[ip2].shortVal*deg2rad;
	  
	  /*	  sla_eqgal_(&val1,&val2,&gl,&gb); */  /* Call SLALIB routine to convert to gl,gb */ 
	  convertGalactic(val1,val2,&gl,&gb);
	  /*	  slaEqgal(val1,val2,&gl,&gb);*/
	  gl/=deg2rad;
	  gb/=deg2rad;
	  psr->param[ip3].set1 = 1; psr->param[ip3].shortVal = gl; sprintf(psr->param[ip3].val,"%-6.2f",gl);        
	  psr->param[ip4].set1 = 1; psr->param[ip4].shortVal = gb; sprintf(psr->param[ip4].val,"%-6.2f",gb);        
	  psr->param[ip3].derived = 1;	  psr->param[ip4].derived = 1;
	  if (pcat_maxSize[ip3] < strlen(psr->param[ip3].val)) pcat_maxSize[ip3]=strlen(psr->param[ip3].val);
	  if (pcat_maxSize[ip4] < strlen(psr->param[ip4].val)) pcat_maxSize[ip4]=strlen(psr->param[ip4].val);
	}
    }
  else if (type==2) /* ZZ */
    {
      ip1  = getParam("GL",psr);
      ip2  = getParam("GB",psr);
      ip3  = getParam("DIST",psr);
      val1 = psr->param[ip1].shortVal*deg2rad;
      val2 = psr->param[ip2].shortVal*deg2rad;

      val3 = psr->param[ip3].shortVal;
      val  = val3*sin(val2);
      psr->param[ip].set1 = 1; psr->param[ip].shortVal = val; sprintf(psr->param[ip].val,"%-6.2f",val);        
      psr->param[ip].derived = 1;
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);
    }
  else if (type==3) /* XX */
    {
      ip1  = getParam("GL",psr);
      ip2  = getParam("GB",psr);
      ip3  = getParam("DIST",psr);
      val1 = psr->param[ip1].shortVal*deg2rad;
      val2 = psr->param[ip2].shortVal*deg2rad;

      val3 = psr->param[ip3].shortVal;
      val  = val3*cos(val2)*sin(val1);
      psr->param[ip].set1 = 1; psr->param[ip].shortVal = val; sprintf(psr->param[ip].val,"%-6.2f",val);        
      psr->param[ip].derived = 1;
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);
    }
  else if (type==4) /* YY */
    {
      ip1  = getParam("GL",psr);
      ip2  = getParam("GB",psr);
      ip3  = getParam("DIST",psr);
      val1 = psr->param[ip1].shortVal*deg2rad;
      val2 = psr->param[ip2].shortVal*deg2rad;

      val3 = psr->param[ip3].shortVal;
      val  = 8.5 - val3*cos(val2)*cos(val1);
      psr->param[ip].set1 = 1; psr->param[ip].shortVal = val; sprintf(psr->param[ip].val,"%-6.2f",val);        
      psr->param[ip].derived = 1;
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);
    }
  else if (type==5) /* DMSINB */
    {
      ip1  = getParam("DM",psr);
      ip2  = getParam("GB",psr);
      if (psr->param[ip1].set1 == 1){
	val1 = psr->param[ip1].shortVal;
	val2 = psr->param[ip2].shortVal*deg2rad;
	
	val = val1*sin(val2);
	psr->param[ip].set1 = 1; psr->param[ip].shortVal = val; sprintf(psr->param[ip].val,"%-6.2f",val);        
	psr->param[ip].derived = 1;
	if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);
      }
    }
}

/* ******************* */
/* Distance parameters */
/* ******************* */
void defineDistance(pulsar *psr,int ip,int type)
{
  int ip1,ip2,ip3,ip4,ip5;
  double val,val1,val2,val3,val4;
  double sigma=-1;
  /* Calculate dist */
  /* -- if dist_a exists then use dist_a */
  /* -- if PX exists then use 1/PX if PX is >3 sigma significance (updated June 3rd 2012: GH)*/
  /* -- if dist_amn and dist_amx exist and dist_dm lies within boundary */
  /*      then use dist_dm else use the closest limit to dist_dm */
  /*  - if dist_dm is not defined then use (dism_amn + dist_amx)/2
  /* -- otherwise use dist_dm */

  ip1 = getParam("PX",NULL);
  ip2 = getParam("DIST_A",NULL);
  ip3 = getParam("DIST_AMN",NULL);
  ip4 = getParam("DIST_AMX",NULL);

  if (type==1)      ip5 = getParam("DIST_DM",NULL);
  else if (type==2) ip5 = getParam("DIST_DM1",NULL);

  if (psr->param[ip1].set1==1) /* Have parallax */
    {
      int ierr;
      sscanf(psr->param[ip1].err,"%d",&ierr);
      sigma = fabs(psr->param[ip1].shortVal/(ierr*psr->param[ip1].error_expand));
    }

  if (psr->param[ip2].set1==1) /* Have DIST_A */
    val  = psr->param[ip2].shortVal;
  else if (sigma > 3) /* Have parallax greater than 3 sigma significance*/
    {
      val1 = psr->param[ip1].shortVal;
      val  = oneAU/(val1*M_PI/1000.0/60.0/60.0/180.0)/onePC/1000.0; /* In kpc */
    }
  else if (psr->param[ip3].set1==1 && psr->param[ip4].set1==1) /* Have DIST_AMX and DIST_AMN */
    {
      val1 = psr->param[ip3].shortVal;
      val2 = psr->param[ip4].shortVal;
      val3 = psr->param[ip5].shortVal;
      /* If dist_tc is within the bounds then use dist_tc, otherwise use bound closest to dist_tc */
      if (psr->param[ip5].set1==1)
	{
	  if (val3 <= val2 && val3 >= val1)
	    val = val3;
	  else
	    {
	      if (val3 >= val2)
		val = val2;
	      else
		val = val1;
	    }
	}
      else
	val = (val1 + val2)/2.0;
    }
  else if (psr->param[ip5].set1==1) /* Have DIST_DM or DIST_DM1*/
    val = psr->param[ip5].shortVal;
  else 
    return;

  psr->param[ip].set1 = 1;
  psr->param[ip].derived = 1;
  psr->param[ip].shortVal = val;
  sprintf(psr->param[ip].val,"%-6.2f",val);        
  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);
    
}

/* ****************************** */
/* BASED ON PERIOD AND DERIVATIVE */
/* ****************************** */
void derive_basedP(pulsar *psr,int ip,int type)
{
  int p0,p1,setP1,ip2,ip3;
  double v1,v2,v3,v4,v6;

  p0 = getParam("P0",NULL);
  p1 = getParam("P1",NULL);
  if ((setP1 = psr->param[p1].set1)!=1)
    return;
  v1 = psr->param[p0].shortVal;
  v2 = psr->param[p1].shortVal;

  if (v2 <= 0) /* Bad value for Pdot */ 
    return;

  if (type==1 && v2>0)      /* Characteristic age (in years) */
    {
      v3 = v1/2.0/v2/60.0/60.0/24.0/365.25;
      psr->param[ip].shortVal = v3;
      sprintf(psr->param[ip].val,"%-.3g",v3);
    }
  else if (type==2 && v2>0) /* Surface magnetic field */
    {
      v3 = 3.2e19*sqrt(v1*v2);
      psr->param[ip].shortVal = v3;
      sprintf(psr->param[ip].val,"%-9.3g",v3);
    }
  else if (type==3 && v2>0) /* Surface magnetic field at light cylinder */
    {
      v3 = 3.0e8*sqrt(v2)*pow(v1,-5.0/2.0);
      psr->param[ip].shortVal = v3;
      sprintf(psr->param[ip].val,"%-6.2e",v3);
    }
  else if (type==4 && v2>0) /* EDOT */
    {
      v3 = 4.0 * M_PI * M_PI * PSR_INERTIA * v2 / pow(v1,3);
      psr->param[ip].shortVal = v3;
      sprintf(psr->param[ip].val,"%-.1e",v3);
    }
  else if (type==5 && v2>0) /* EDOTD2 */
    {
      ip2 = getParam("DIST",psr);
      v4 = psr->param[ip2].shortVal;

      // j.khoo: 13.4.10
      // only calculate EDOTD2 if values for both DIST AND EDOT exist
      if (v4 != 0.0) {

        /*      v3 = PSR_INERTIA/3.086e21/3.086e21 * 4.0 * (M_PI * M_PI) * 
                v2 / 4.0 / M_PI / pow(v1,3.0)/(v4*v4);  */
        /* In units of erg kpc-2 s-1 */
        v3 = PSR_INERTIA * 4.0 * M_PI * M_PI * v2 / pow(v1,3.0)/(v4*v4);
        /*      printf("Have %g %g\n",v4,v3);
                exit(1); */

        psr->param[ip].shortVal = v3;
        sprintf(psr->param[ip].val,"%-.1e",v3);
      } else {
        return;
      }
    }
  
  psr->param[ip].set1 = 1;
  psr->param[ip].derived = 1;
  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) pcat_maxSize[ip]=strlen(psr->param[ip].val);



}

/* ********************* */
/* Define RAJD and DECJD */
/* ********************* */
void defineDegrees(pulsar *psr,int ip,int type)
{
  double val;  
  int    ip_raj,ip_decj;

  if (psr->param[ip].set1==0)
    {
      if (type==1) ip_raj  = getParam("RAJ",NULL);
      if (type==2) ip_decj = getParam("DECJ",NULL);

      if (type==1) psr->param[ip].shortVal = val = turn_deg(hms_turn(psr->param[ip_raj].val));
      if (type==2) psr->param[ip].shortVal = val = turn_deg(dms_turn(psr->param[ip_decj].val));
      sprintf(psr->param[ip].val,"%-10.5f",val);
      psr->param[ip].set1 = 1;
      psr->param[ip].derived = 1;
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) 
	pcat_maxSize[ip]=strlen(psr->param[ip].val);
    }  
}

/* ************************ */
/* Define Binary parameters */
/* ************************ */
void defineBinary(pulsar *psr,int ip,int type)
{
  double val,gm=1.3271243999e26,asini,pb,eps1,eps2,err1,err2,err,val3;
  double day2sec = 86400.0,speedlight=2.99792458e10,massfn,val2,val1;
  int lv,le;
  char msg[MAX_STRLEN];
  int ip1,ierr_eps1,ierr_eps2;
  int ip2,ip3;

  if (psr->param[ip].set1==0)
    {     
      if (type==1 || type==2 || type==3 || type==4) /* MASSFN/MINMASS/MEDMASS/UPMASS */
	{
	  ip1 = getParam("A1",NULL);
	  ip2 = getParam("PB",NULL);
	  if (psr->param[ip1].set1==1 && psr->param[ip2].set1==1)
	    {
	      pb    = psr->param[ip2].shortVal*day2sec;
	      asini = psr->param[ip1].shortVal*speedlight; 
	      massfn = 2.0*M_PI*2.0*M_PI/gm*pow(asini,3)/pb/pb;

	      if (type==1) /* MASSFN */
		{
		  psr->param[ip].shortVal = val = massfn;
		  sprintf(psr->param[ip].val,"%10.6f",val);
		}
	      else if (type==2) /* MINMASS */
		{
		  psr->param[ip].shortVal = val = m2(massfn,1.0,MASS_PSR);
		  sprintf(psr->param[ip].val,"%10.6f",val);
		}
	      else if (type==3) /* MEDMASS */
		{
		  psr->param[ip].shortVal = val = m2(massfn,0.866025403,MASS_PSR);
		  sprintf(psr->param[ip].val,"%10.6f",val);
		}
	      else if (type==4) /* UPRMASS */
		{
		  psr->param[ip].shortVal = val = m2(massfn,0.438371146,MASS_PSR);
		  sprintf(psr->param[ip].val,"%10.6f",val);
		}
	      psr->param[ip].set1 = 1;
	      psr->param[ip].derived = 1;
	      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) 
		pcat_maxSize[ip]=strlen(psr->param[ip].val);	      
	    }
	}
      if (type==5 || type==6) /* OM/ECC from EPS1 and EPS2 if ELL1 model */
	{
	  ip1 = getParam("EPS1",NULL);
	  ip2 = getParam("EPS2",NULL);
	  if (psr->param[ip1].set1==1 && psr->param[ip2].set2==1)
	    {
	      eps1 = psr->param[ip1].shortVal;
	      eps2 = psr->param[ip2].shortVal;
	      sscanf(psr->param[ip1].err,"%d",&ierr_eps1);
	      sscanf(psr->param[ip2].err,"%d",&ierr_eps2);	      
	      err1 = psr->param[ip1].error_expand*ierr_eps1;
	      err2 = psr->param[ip2].error_expand*ierr_eps2;
	      val2 = sqrt(pow(eps1,2)+pow(eps2,2)); /* ECC */
	      if (type==5 && val2 != 0.0) /* OM */
		{
		  val  = atan2(eps1,eps2)*180.0/M_PI;
		  err  = pow(pow(eps2*err1,2)+pow(eps1*err2,2),0.5)/val2/val2*180.0/M_PI;		  
		  if (val<0) val+=360.0;
		  rnd8(val,err,1,psr->param[ip].val,&lv,psr->param[ip].err,&le,msg);
		  psr->param[ip].shortVal = val;
		}
	      else if (type==6 && val2!=0.0) /* ECC */
		{
		  val = val2;
		  /* What should this be if EPS1 and EPS2 = 0 */
		  if (eps1==0.0 && eps2==0.0)
		    err = sqrt(pow(err1,2)+pow(err2,2));
		  else
		    err = pow(pow(eps1*err1,2)+pow(eps2*err2,2),0.5)/val;
		  rnd8(val,err,1,psr->param[ip].val,&lv,psr->param[ip].err,&le,msg);
		  psr->param[ip].shortVal = val;		  
		}
	      if (val2!=0.0)
		{
		  strcpy(psr->param[ip].ref,psr->param[ip1].ref); 
		  
		  psr->param[ip].set1 = 1;
		  psr->param[ip].set2 = 1;
		  psr->param[ip].set3 = 1;
		  psr->param[ip].derived = 1;
		  if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) 
		    pcat_maxSize[ip]=strlen(psr->param[ip].val);	      
		}
	      else
		psr->param[ip].set1 = 0;
	    }
	  else
	    psr->param[ip].set1 = 0;
	}
      if (type==7) /* MINOMDOT */
	{
	  ip1  = getParam("ECC",psr);
	  ip2 =  getParam("PB",NULL);
	  ip3 =  getParam("MINMASS",psr);
	  if (psr->param[ip1].set1==1 && psr->param[ip2].set1==1 && psr->param[ip3].set1==1)
	    {
	      val1 = psr->param[ip1].shortVal; /* ECC */
	      val2 = psr->param[ip2].shortVal*day2sec; /* PB  */
	      val3 = psr->param[ip3].shortVal; /* MINMASS  */
	      val = 3.0*		
		pow(2.0*M_PI/val2*day2sec/86400.0, 5.0/3.0)*
		pow((MASS_PSR+val3)*4.925490946e-6, 2.0/3.0)/
		(1.0-val1*val1)*
		180.0/M_PI*          /* rad/sec --> deg/sec */
		86400.0*365.25;      /* deg/sec --> deg/yr */	      
	      sprintf(psr->param[ip].val,"%10.3g",val);	      
	      psr->param[ip].set1 = 1;
	      psr->param[ip].derived = 1;
	      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) 
		pcat_maxSize[ip]=strlen(psr->param[ip].val);	      
	    }
	}
    }  
}

/* ************************ */
/* Derive custom parameters */
/* ************************ */
void defineCustom(pulsar *psr,int ip,int type,int exactMatch)
{
  double v1;
  int badval;

  if (strlen(pcat_custom1)<10 && type==1)
    { printf("Custom parameter 1 not defined\n"); exit(1); } 
  else if (strlen(pcat_custom2)<10 && type==2)
    { printf("Custom parameter 2 not defined\n"); exit(1); } 
  else if (strlen(pcat_custom3)<10 && type==3)
    { printf("Custom parameter 3 not defined\n"); exit(1); } 
  else if (strlen(pcat_custom4)<10 && type==4)
    { printf("Custom parameter 4 not defined\n"); exit(1); } 

  psr->param[ip].set1 = 0;
  
  if (type==1) psr->param[ip].shortVal = v1 = formExpression(pcat_custom1,psr,&badval,exactMatch);
  if (type==2) psr->param[ip].shortVal = v1 = formExpression(pcat_custom2,psr,&badval,exactMatch);
  if (type==3) psr->param[ip].shortVal = v1 = formExpression(pcat_custom3,psr,&badval,exactMatch);
  if (type==4) psr->param[ip].shortVal = v1 = formExpression(pcat_custom4,psr,&badval,exactMatch);
  if (badval==0)
    {
      sprintf(psr->param[ip].val,CUSTOM_FORMAT,v1);
      psr->param[ip].set1 = 1;
      psr->param[ip].derived = 1;
      if (pcat_maxSize[ip] < strlen(psr->param[ip].val)) 
	pcat_maxSize[ip]=strlen(psr->param[ip].val);
    }
}

double turn_deg(double turn){
 
  /* Converts double turn to string "sddd.ddd" */
  return turn*360.0;
}

int turn_dms(double turn, char *dms){
  
  /* Converts double turn to string "sddd:mm:ss.sss" */
  
  int dd, mm, isec;
  double trn, sec;
  char sign;
  
  sign=' ';
  if (turn < 0.){
    sign = '-';
    trn = -turn;
  }
  else{
    sign = '+';
    trn = turn;
  }
  dd = trn*360.;
  mm = (trn*360.-dd)*60.;
  sec = ((trn*360.-dd)*60.-mm)*60.;
  isec = (sec*1000. +0.5)/1000;
    if(isec==60){
      sec=0.;
      mm=mm+1;
      if(mm==60){
        mm=0;
        dd=dd+1;
      }
    }
  sprintf(dms,"%c%02d:%02d:%010.7f",sign,dd,mm,sec);
 
}


int turn_hms(double turn, char *hms){
 
  /* Converts double turn to string " hh:mm:ss.ssss" */
  
  int hh, mm, isec;
  double sec;

  hh = turn*24.;
  mm = (turn*24.-hh)*60.;
  sec = ((turn*24.-hh)*60.-mm)*60.;
  isec = (sec*10000. +0.5)/10000;
    if(isec==60){
      sec=0.;
      mm=mm+1;
      if(mm==60){
        mm=0;
        hh=hh+1;
        if(hh==24){
          hh=0;
        }
      }
    }

  sprintf(hms," %02d:%02d:%010.7f",hh,mm,sec);
 
}


double hms_turn(char *line){

  /* Converts string " hh:mm:ss.ss" or " hh mm ss.ss" to double turn */
  
  int i;int turn_hms(double turn, char *hms);
  double hr, min, sec, turn=0;
  char hold[MAX_STRLEN];

  strcpy(hold,line);

  /* Get rid of ":" */
  for(i=0; *(line+i) != '\0'; i++)if(*(line+i) == ':')*(line+i) = ' ';

  i = sscanf(line,"%lf %lf %lf", &hr, &min, &sec);
  if(i > 0){
    turn = hr/24.;
    if(i > 1)turn += min/1440.;
    if(i > 2)turn += sec/86400.;
  }
  if(i == 0 || i > 3)turn = 1.0;


  strcpy(line,hold);

  return turn;
}

double dms_turn(char *line){

  /* Converts string "-dd:mm:ss.ss" or " -dd mm ss.ss" to double turn */
  
  int i;
  char *ic, ln[40];
  double deg, min, sec, sign, turn=0;

  /* Copy line to internal string */
  strcpy(ln,line);

  /* Get rid of ":" */
  for(i=0; *(ln+i) != '\0'; i++)if(*(ln+i) == ':')*(ln+i) = ' ';

  /* Get sign */
  if((ic = strchr(ln,'-')) == NULL)
     sign = 1.;
  else {
     *ic = ' ';
     sign = -1.;
  }

  /* Get value */
  i = sscanf(ln,"%lf %lf %lf", &deg, &min, &sec);
  if(i > 0){
    turn = deg/360.;
    if(i > 1)turn += min/21600.;
    if(i > 2)turn += sec/1296000.;
    if(turn >= 1.0)turn = turn - 1.0;
    turn *= sign;
  }
  if(i == 0 || i > 3)turn =1.0;

  return turn;
}

/* ************************************************************************
   m2 - solves mass function for m2, using the Newton-Raphson method

   where:  mf = mass function
           m1 = primary mass
           si = sin(i) (i = inclination angle)

   solves: (m1+m2)^2 = (m2*si)^3 / mf

   returns -1 on error

   WVS Jan 2000
************************************************************************ */

double m2 (double mf, double sini, double m1)
{
  double guess = m1;
  double dx = 0.0;
  double eq = 0.0;
  double deq_dm2 = 0.0;
  int gi = 0;

  for (gi=0; gi<10000; gi++) {
    eq = pow(m1+guess,2) - pow(guess*sini,3) / mf;
    deq_dm2 = 2.0*(m1+guess) - 3.0 * pow(guess*sini,2) / mf;

    dx = eq / deq_dm2;
    guess -= dx;

    if (fabs (dx) <= fabs(guess)*1e-10)
      return guess;
  }
  fprintf (stdout,"m2: maximum iterations exceeded - %lf\n", fabs(dx/guess));
  return -1.0;
}


 /* the survey names - the first name corresponds to bit 1 being set */
 /* in the survey filed of the spulsar structure. */

int encode_survey( char * inptr,int survey_num,char surveynames[MAX_STRLEN][MAX_SURVEYS])
{
  unsigned int temp=0, imask=0;
  int i,found;
  char * survnam=NULL;

  /* Modified by G. Hobbs (13/01/03), Surveys now deliminated by ',' not white space */

  for(survnam=strtok(inptr,",\n"); survnam != NULL; 
      survnam=strtok(NULL,",\n")) 
    {
      imask = 1;
      i = 0;
      found = 0;
      while( i <= survey_num && strcmp(survnam,surveynames[i++]) != 0 ) 
	imask <<= 1;
      if (strcmp(survnam,surveynames[i-1])==0)
	found=1;
      if( found==0)
	{ 
	  printf("unknown survey : %s (please contact the ATNF if this survey should be entered into the psrcat survey list)\n",survnam);
	}
      else
	{
	  temp |= imask;
	}
    }

/*	printf("%s : %o : %d/%d\n", inptr, temp, i, NSURV);*/

  return temp;
}



/* Surveys */
int define_surveys(char surveynames[MAX_STRLEN][MAX_SURVEYS])
{
  int survey_num;

  strcpy(surveynames[0],"mol1");
  strcpy(surveynames[1],"jb1");
  strcpy(surveynames[2],"ar1");
  strcpy(surveynames[3],"mol2");
  strcpy(surveynames[4],"gb1");
  strcpy(surveynames[5],"gb2");
  strcpy(surveynames[6],"jb2");
  strcpy(surveynames[7],"gb3");
  strcpy(surveynames[8],"ar2");
  strcpy(surveynames[9],"pks1");
  strcpy(surveynames[10],"ar3");
  strcpy(surveynames[11],"pks70");
  strcpy(surveynames[12],"gb4");
  strcpy(surveynames[13],"ar4");
  strcpy(surveynames[14],"pksmb");
  strcpy(surveynames[15],"pkssw");
  strcpy(surveynames[16],"pksgc");
  strcpy(surveynames[17],"misc");
  strcpy(surveynames[18],"pkshl");
  strcpy(surveynames[19],"palfa");
  strcpy(surveynames[20],"FermiBlind");
  strcpy(surveynames[21],"htru_pks");
  strcpy(surveynames[22],"FermiAssoc");
  strcpy(surveynames[23],"gb350");
  strcpy(surveynames[24],"pkspa");
  strcpy(surveynames[25],"pksngp");
  strcpy(surveynames[26],"htru_eff");
  strcpy(surveynames[27],"ar327");
  strcpy(surveynames[28],"gbncc");
  strcpy(surveynames[29],"ghrss");
  
  survey_num=29;
  return survey_num;
}

double galpa(double el, double eb){
  /* Returns Equatorial PA of ecliptic meridian, given (l,b) (degrees) */

  float  fgl, fgb, fglpole, fgbpole;
  double eclpole, ecbpole;
  double dgpa;
  double date,v1,v2; 
  double first,second;

  date = 51544.0;

  el = el*2.0*M_PI/360.0;       /* Convert to radians */
  eb = eb*2.0*M_PI/360.0;

  /* Galactic coords of J2000 pole */
  first = 0.0;
  second = M_PI/4.;

  /*  sla_eqecl_(&first,&second,&date,&eclpole,&ecbpole); */
  /*  slaEqecl(0.0, DPI/4., date, &eclpole, &ecbpole);*/
  convertEcliptic(0.0,M_PI/4.0,&eclpole,&ecbpole);
  v1 = 0.0;
  v2 = 2.0*M_PI/4.0;
  /*  sla_eqecl_(&v1,&v2,&date,&eclpole,&ecbpole); */
  /* Get ecliptic PA */
  fgl = el, fgb = eb, fglpole = eclpole, fgbpole = ecbpole;
  dgpa = posAngle(fgl,fgb,fglpole,fgbpole);   
  /*  dgpa = sla_bear_(&fgl,&fgb,&fglpole,&fgbpole);  */
  dgpa = dgpa/(2.0*M_PI)*360.0;            /* Convert to degrees */
 
  return -dgpa;
}

void defineParameters(paramtype *defParam)
{
  addParameter(defParam,"PSRJ","%-12.12s","PSRJ"," ","Pulsar Jname",1,1);
  addParameter(defParam,"PSRB","%-12.12s","PSRB"," ","Pulsar Bname",1,0);
  addParameter(defParam,"NAME","%-12.12s","NAME"," ","Pulsar name",1,0);
  addParameter(defParam,"RAJ","%-10.10s","RAJ","(hms)","Right Ascension (J2000) (hh:mm:ss.s)",1,1);
  addParameter(defParam,"DECJ","%-11.11s","DECJ","(dms)","Declination (J2000) (+dd:mm:ss)",1,1);
  addParameter(defParam,"ELONG","%9.3f","ELONG","(deg)","Ecliptic longitude (deg)",1,1);
  addParameter(defParam,"ELAT","%9.3f","ELAT","(deg)","Ecliptic latitude (deg)",1,1);
  addParameter(defParam,"DM","%10.2f","DM","(cm^-3 pc)","Dispersion measure (cm**-3 pc)",2,1);
  addParameter(defParam,"PEPOCH","%8.2f","PEPOCH","(MJD)","Epoch of period/frequency determination (MJD)",2,1);
  addParameter(defParam,"F0","%12.6f","F0","(Hz)","Barycentric rotation frequency (Hz)",2,1);
  addParameter(defParam,"F1","%10.3e","F1","(s^-2)","Time deriv of baryctr rotn frequency (s^-2)",2,1);
  addParameter(defParam,"P0","%9.6f","P0","(s)","Barycentric period of the pulsar (sec)",2,0);
  addParameter(defParam,"P1","%8.2e","P1"," ","Period derivative",2,0);
  addParameter(defParam,"DIST_DM","%9.2f","DIST_DM","(kpc)","Distance based on tc93 electron density model",4,0);
  addParameter(defParam,"DIST_DM1","%9.2f","DIST_DM1","(kpc)","Distance based on tc93 electron density model",4,0);
  addParameter(defParam,"SURVEY","%-35.35s","SURVEY"," ","Surveys that detected pulsar (discovery survey first)",5,0);
  addParameter(defParam,"PMRA","%9.3f","PMRA","(mas/yr)","Proper motion in RA (mas/yr)",1,1);
  addParameter(defParam,"PMDEC","%9.3f","PMDEC","(mas/yr)","Proper motion in DEC (mas/yr)",1,1);
  addParameter(defParam,"S30","%8.2f","S30","(mJy)","Flux at 30MHz (mJy)",2,0);
  addParameter(defParam,"S40","%8.2f","S40","(mJy)","Flux at 40MHz (mJy)",2,0);
  addParameter(defParam,"S50","%8.2f","S50","(mJy)","Flux at 50MHz (mJy)",2,0);
  addParameter(defParam,"S60","%8.2f","S60","(mJy)","Flux at 60MHz (mJy)",2,0);
  addParameter(defParam,"S80","%8.2f","S80","(mJy)","Flux at 80MHz (mJy)",2,0);
  addParameter(defParam,"S100","%8.2f","S100","(mJy)","Flux at 100MHz (mJy)",2,0);
  addParameter(defParam,"S150","%8.2f","S150","(mJy)","Flux at 150MHz (mJy)",2,0);
  addParameter(defParam,"S200","%8.2f","S200","(mJy)","Flux at 200MHz (mJy)",2,0);
  addParameter(defParam,"S300","%8.2f","S300","(mJy)","Flux at 300MHz (mJy)",2,0);
  addParameter(defParam,"S400","%8.2f","S400","(mJy)","Flux at 400MHz (mJy)",2,0);
  addParameter(defParam,"S600","%8.2f","S600","(mJy)","Flux at 600MHz (mJy)",2,0);
  addParameter(defParam,"S800","%8.2f","S800","(mJy)","Flux at 800MHz (mJy)",2,0);
  addParameter(defParam,"S1400","%8.2f","S1400","(mJy)","Flux at 1400MHz (mJy)",2,0);
  addParameter(defParam,"S2000","%8.2f","S2000","(mJy)","Flux at 2000MHz (mJy)",2,0);
  addParameter(defParam,"S4000","%8.2f","S4000","(mJy)","Flux at 4000MHz (mJy)",2,0);
  addParameter(defParam,"S5000","%8.2f","S5000","(mJy)","Flux at 5000MHz (mJy)",2,0);
  addParameter(defParam,"S6000","%8.2f","S6000","(mJy)","Flux at 6000MHz (mJy)",2,0);
  addParameter(defParam,"S8000","%8.2f","S8000","(mJy)","Flux at 8000MHz (mJy)",2,0);
  addParameter(defParam,"S10G","%8.2f","S10G","(mJy)","Flux at 10GHz (mJy)",2,0);
  addParameter(defParam,"S20G","%8.2f","S20G","(mJy)","Flux at 20GHz (mJy)",2,0);
  addParameter(defParam,"S50G","%8.2f","S50G","(mJy)","Flux at 50GHz (mJy)",2,0);
  addParameter(defParam,"S100G","%8.2f","S100G","(mJy)","Flux at 100GHz (mJy)",2,0);
  addParameter(defParam,"S150G","%8.2f","S150G","(mJy)","Flux at 150GHz (mJy)",2,0);
  addParameter(defParam,"W50","%8.3f","W50","(ms)","Width of pulse at 50% (msec)",2,0);
  addParameter(defParam,"W10","%8.3f","W10","(ms)","Width of pulse at 10% (msec)",2,0);
  addParameter(defParam,"POSEPOCH","%8.2f","POSEPOCH","(MJD)","Epoch at which the position is measured (MJD)",1,1);
  addParameter(defParam,"DMEPOCH","%8.2f","DMEPOCH","(MJD)","Epoch at which the dispersion measure is measured (MJD)",1,1);
  addParameter(defParam,"F2","%10.3e","F2","(s^-3)","Second time deriv of baryctr rotn frequency (s^-3)",2,1);
  addParameter(defParam,"BINARY","%-8.8s","BINARY","(type)","Binary model",3,1);
  addParameter(defParam,"PB","%10.4f","PB","(days)","Binary period of pulsar (days)",3,1);
  addParameter(defParam,"ECC","%8.3e","ECC"," ","Eccentricity",3,1);
  addParameter(defParam,"A1","%10.4f","A1","(lt sec)","Projected semi-major axis of orbit (lt sec)",3,1);
  addParameter(defParam,"T0","%10.4f","T0","(MJD)","Epoch of periastron (MJD)",3,1);
  addParameter(defParam,"OM","%8.2f","OM","(deg)","Longitude of periastron (degrees)",3,1);
  addParameter(defParam,"OMDOT","%10.5f","OMDOT","(deg/yr)","Periastron advance (degrees/yr)",3,1);
  addParameter(defParam,"TASC","%10.4f","TASC","(MJD)","Epoch of ascending node (MJD)",3,1);
  addParameter(defParam,"TASC_2","%10.4f","TASC_2","(MJD)","Epoch of ascending node (MJD) [2nd orbit]",3,1);
  addParameter(defParam,"EPS1","%10.3e","EPS1"," ","ECC*sin(OM) - ELL1 binary model",3,1);
  addParameter(defParam,"EPS1_2","%10.3e","EPS1_2"," ","ECC*sin(OM) - ELL1 binary model [2nd orbit]",3,1);
  addParameter(defParam,"EPS2","%10.3e","EPS2"," ","ECC*cos(OM) - ELL1 binary model",3,1);
  addParameter(defParam,"EPS2_2","%10.3e","EPS2_2"," ","ECC*cos(OM) - ELL1 binary model [2nd orbit]",3,1);
  addParameter(defParam,"OM_ASC","%8.2f","OM_ASC","(deg)","Position angle of ascending node (deg)",3,1);
  addParameter(defParam,"OM_ASC_2","%8.2f","OM_ASC_2","(deg)","Position angle of ascending node (deg) [second orbit]",3,1);
  addParameter(defParam,"START","%8.2f","START","(MJD)","Epoch of start of fit (MJD)",7,1);
  addParameter(defParam,"FINISH","%8.2f","FINISH","(MJD)","Epoch of end of fit (MJD)",7,1);
  addParameter(defParam,"TRES","%8.2f","TRES","(us)","RMS timing residual (microsec)",7,0);
  addParameter(defParam,"CLK","%8s","CLK"," ","Terrestrial time standard",7,1);
  addParameter(defParam,"DIST_AMN","%8.2f","DIST_AMN","(kpc)","Lower limit on alternate distance based on, e.g., association or HI absorption (kpc)",4,0);
  addParameter(defParam,"DIST_AMX","%8.2f","DIST_AMX","(kpc)","Upper limit on alternate distance based on, e.g., association or HI absorption (kpc)",4,0);
  addParameter(defParam,"DIST_A","%8.2f","DIST_A","(kpc)","Alternate distance, e.g., from association or HI absorption (kpc)",4,0);
  addParameter(defParam,"ASSOC","%-40.40s","ASSOC"," ","Association with SNR, GC etc.",5,0);
  addParameter(defParam,"BINCOMP","%-35.35s","BINCOMP"," ","Binary companion type",5,0);
  addParameter(defParam,"AGE","%6.2e","AGE","(Yr)","Spin down age (yr)",6,0);
  addParameter(defParam,"C1","%16.7g","C1"," ","Custom parameter 1",11,0);
  addParameter(defParam,"C2","%16.7g","C2"," ","Custom parameter 2",11,0);
  addParameter(defParam,"C3","%16.7g","C3"," ","Custom parameter 3",11,0);
  addParameter(defParam,"C4","%16.7g","C4"," ","Custom parameter 4",11,0);
  addParameter(defParam,"EPHEM","%8s","EPHEM"," ","Solar system ephemeris",7,1);
  addParameter(defParam,"PBDOT","%10.5f","PBDOT"," ","First time derivative of binary period",3,1);
  addParameter(defParam,"SPINDX","%6.2f","SPINDX"," ","Measured spectral index",2,0);
  addParameter(defParam,"TZRMJD","%20.14f","TZRMJD","(MJD)","Reference TOA (MJD)",7,1);
  addParameter(defParam,"TZRFRQ","%10.3f","TZRFRQ","(MHz)","Frequency of reference TOA (MHz)",7,1);
  addParameter(defParam,"TZRSITE","%-8.8s","TZRSITE"," ","One-letter observatory code of reference TOA",7,1);
  addParameter(defParam,"RM","%10.2f","RM","(rad m^-2)","Rotation measure (rad m**-2)",2,1);
  addParameter(defParam,"TAU_SC","%6.2f","TAU_SC","(s)","Temporal broadening of pulses at 1 GHz (s)",2,0);
  addParameter(defParam,"S900","%8.2f","S900","(mJy)","Flux at 900MHz (mJy)",2,0);
  addParameter(defParam,"S700","%8.2f","S700","(mJy)","Flux at 700MHz (mJy)",2,0);
  addParameter(defParam,"S3000","%8.2f","S3000","(mJy)","Flux at 3000MHz (mJy)",2,0);
  addParameter(defParam,"PX","%8.3f","PX","(mas)","Annual parallax (mas)",1,1);
  addParameter(defParam,"TYPE","%-25.25s","PSR","TYPE","Type codes for  pulsar",5,0);
  addParameter(defParam,"NTOA","%5d","NTOA"," ","Number of TOAs in timing fit",7,1);
  addParameter(defParam,"S1600","%8.2f","S1600","(mJy)","Flux at 1600MHz (mJy)",2,0);
  addParameter(defParam,"NGLT","%8.0f","NGLT"," ","Number of observed glitches",12,0);
  addParameter(defParam,"ECCDOT","%10.5f","ECCDOT"," ","First time derivative of eccentricity",3,1);
  addParameter(defParam,"A1DOT","%10.5f","A1DOT"," ","First time derivative of projected semi-major axis (lt-s s^-1)",3,1);
  addParameter(defParam,"T0_2","%10.5f","T0_2","(MJD)","Epoch of periastron [2nd orbit] (MJD)",3,1);
  addParameter(defParam,"PB_2","%10.5f","PB_2","(days)","Binary period of pulsar [2nd orbit]  (days)",3,1);
  addParameter(defParam,"FB0","%9.3e","FB0","(Hz)","Binary frequency of pulsar 1/(PB*86400) (Hz)",3,1);
  addParameter(defParam,"FB1","%9.3e","FB1","(Hz s^-1)","First time derivative of binary frequency (Hz s^-1)",3,1);
  addParameter(defParam,"FB2","%9.3e","FB2","(Hz s^-2)","Second time derivative of binary frequency (Hz s^-2)",3,1);
  addParameter(defParam,"A12DOT","%10.5f","A12DOT","(lt-s s^-2)","Second time derivative of projected semi-major axis (lt-s s^-2)",3,1);
  addParameter(defParam,"OM2DOT","%10.5f","OM2DOT","(deg yr^-2)","Second time derivative of longitude of periastron (deg yr^-2)",3,1);
  addParameter(defParam,"ECC2DOT","%10.5f","ECC2DOT","(s^-2)","Second time derivative of eccentricity (s^-2)",3,1);

  addParameter(defParam,"A1_2","%10.5f","A1_2"," ","Projected semi-major axis of orbit [2nd orbit] (sec)",3,1);
  addParameter(defParam,"ECC_2","%10.5f","ECC_2"," ","Eccentricity [2nd orbit]",3,1);
  addParameter(defParam,"OM_2","%10.5f","OM_2","(deg)","Longitude of periastron [2nd orbit] (degrees)",3,1);
  addParameter(defParam,"DM1","%13.3f","DM1","(cm^-3 pc/yr)","First time derivative of dispersion measure (cm**-3 pc yr^-1)",2,1);
  addParameter(defParam,"F3","%10.3e","F3","(s^-4)","Third time deriv of baryctr rotn frequency (s^-4)",2,1);
  addParameter(defParam,"F4","%10.3e","F4","(s^-5)","Fourth time deriv baryctr rotn frequency (s^-5)",2,1);
  addParameter(defParam,"F5","%10.3e","F5","(s^-6)","Fifth time deriv baryctr rotn frequency (s^-6)",2,1);
  addParameter(defParam,"F6","%10.3e","F6","(s^-7)","Sixth time deriv baryctr rotn frequency (s^-7)",2,1);
  addParameter(defParam,"PMELONG","%9.3f","PMELONG","(mas/yr)","Proper motion in Ecliptic longitude (mas/yr)",1,1);
  addParameter(defParam,"DM2","%15.3f","DM2","(cm^-3 pc yr^-2)","Second time deriv of dispersion measure (cm**-3 pc yr^-2)",2,1);
  addParameter(defParam,"T0_3","%10.5f","T0_3","(MJD)","Epoch of periastron [3rd orbit] (MJD)",3,1);
  addParameter(defParam,"PB_3","%10.5f","PB_3","(days)","Binary period of pulsar [3rd orbit]  (days)",3,1);
  addParameter(defParam,"A1_3","%10.5f","A1_3"," ","Projected semi-major axis of orbit [3nd orbit] (lt sec)",3,1);
  addParameter(defParam,"OM_3","%10.5f","OM_3","(deg)","longitude of periastron [3rd orbit] (degrees)",3,1);
  addParameter(defParam,"GAMMA","%10.5f","GAMMA","(sec)","post-Keplerian 'gamma' term  (sec)",3,1);
  addParameter(defParam,"XPBDOT","%10.5f","XPBDOT"," ","Rate of change of orbital period minus GR prediction",3,1);
  addParameter(defParam,"MTOT","%10.5f","MTOT","(Msun)","Total system mass (solar masses)",3,1);
  addParameter(defParam,"ECC_3","%10.5f","ECC_3"," ","Eccentricity [3rd orbit]",3,1);
  addParameter(defParam,"ALIAS","%-s","ALIAS"," ","Pulsar aliases",1,0);
  addParameter(defParam,"DATE","%4.0f","DISC.","DATE","Year of discovery publication",5,0);
  addParameter(defParam,"DIST","%8.2f","DIST","(kpc)","Best estimate of pulsar distance (kpc)",4,0);  
  addParameter(defParam,"DIST1","%8.2f","DIST1","(kpc)","Best estimate of pulsar distance, using NE2001 model (kpc)",4,0);
  addParameter(defParam,"PMERR_PA","%9.3f","PMERR_PA","(deg)","Angle for proper motion error ellipse",1,0);
  addParameter(defParam,"BRK_INDX","%10.5f","BRK_INDX"," ","Braking index (observed)",2,0);
  addParameter(defParam,"P1_I","%8.2e","P1_I"," ","Period derivative corrected for proper motion effect",2,0);
  addParameter(defParam,"AGE_I","%6.2e","AGE_I","(Yr)","Spin down age from P1_I (yr)",6,0);
  addParameter(defParam,"BSURF_I","%9.2e","BSURF_I","(G)","Surface magnetic dipole from P1_I (gauss)",6,0);
  addParameter(defParam,"EDOT_I","%8.2e","EDOT_I","(ergs/s)","Spin down energy loss rate from P1_I (ergs/s)",6,0);
  addParameter(defParam,"EDOTD2","%8.2e","EDOTD2","(ergs/kpc^2/s)","Energy flux at the Sun (ergs/kpc^2/s)",6,0);
  addParameter(defParam,"R_LUM","%11.2e","R_LUM","(mJy kpc^2)","Radio luminosity at 400MHz (mJy kpc**2)",6,0);
  addParameter(defParam,"R_LUM14","%11.2e","R_LUM14","(mJy kpc^2)","Radio luminosity at 1400MHz (mJy kpc**2)",6,0);
  addParameter(defParam,"PMTOT","%9.3f","PMTOT","(mas/yr)","Total proper motion (mas/yr)",6,0);
  addParameter(defParam,"VTRANS","%9.3f","VTRANS","(km/s)","Transverse velocity - based on DIST (km/s)",6,0);
  addParameter(defParam,"BSURF","%9.2e","BSURF","(G)","Surface magnetic flux density (Gauss)",6,0);
  addParameter(defParam,"B_LC","%9.2e","B_LC","(G)","Magnetic flux density at light cylinder (Gauss)",6,0);
  addParameter(defParam,"SI414","%6.2f","SI414"," ","Spectral index based on S400 & S1400",6,0);
  addParameter(defParam,"EDOT","%8.2e","EDOT","(ergs/s)","Spin down energy loss rate (ergs/s)",6,0);
  addParameter(defParam,"RADDIST","%8.2f","RAD.","DISTANCE","Radial distance from given coordinate (degrees)",4,0);
  addParameter(defParam,"RAJD","%12.6f","RAJD","(deg)","Right Ascension (J2000) (deg)",1,0);
  addParameter(defParam,"DECJD","%12.6f","DECJD","(deg)","Declination (J2000) (deg)",1,0);
  addParameter(defParam,"OSURVEY","%10.10s","OSURVEY"," ","Surveys that detected pulsar encoded as bits in integer",5,0);
  addParameter(defParam,"MASSFN","%7.4f","Mass fn","(Msun)","The pulsar Mass function (M_sun)",3,0);
  addParameter(defParam,"MINMASS","%11.4f","Minimum  ","Mass (Msun)","Minimum companion mass assuming i=90deg and M_ns is 1.35Mo (M_sun)",3,0);
  addParameter(defParam,"MEDMASS","%11.4f","Median  ","Mass (Msun)","Median companion mass assuming i=60deg (M_sun)",3,0);
  addParameter(defParam,"DMSINB","%10.2f","DM*sin(b)","(cm^-3 pc)","DM*sin(b) (cm**-3 pc)",4,0);
  addParameter(defParam,"GL","%9.3f","Gl","(deg)","Galactic Longitude (deg) ",1,0);
  addParameter(defParam,"GB","%9.3f","Gb","(deg)","Galactic Latitude (deg)",1,0);
  addParameter(defParam,"PMELAT","%9.3f","PMELAT","(mas/yr)","Proper motion in Ecliptic latitude (mas/yr)",1,1);
  addParameter(defParam,"XX","%9.2f","XX","(kpc)","X-Distance in X-Y-Z Galactic coordinate system (kpc)",4,0);
  addParameter(defParam,"YY","%9.2f","YY","(kpc)","Y-Distance in X-Y-Z Galactic coordinate system (kpc)",4,0);
  addParameter(defParam,"ZZ","%9.2f","ZZ","(kpc)","Distance from the Galactic plane - based on DIST (kpc)",4,0);
  addParameter(defParam,"INTERIM","%-7.7s","INTERIM"," ","Interim parameters only",5,0);
  addParameter(defParam,"PML","%9.3f","PML      ","(deg/yr)","Proper motion in l (mas/yr)",1,0);
  addParameter(defParam,"PMB","%9.3f","PMB      ","(deg/yr)","Proper motion in b (mas/yr)",1,0);
  addParameter(defParam,"F7","%10.3e","F7","(s^-8)","Seventh time deriv baryctr rotn frequency (s^-8)",2,1);
  addParameter(defParam,"F8","%10.3e","F8","(s^-9)","Eighth time deriv baryctr rotn frequency (s^-9)",2,1);
  addParameter(defParam,"F9","%10.3e","F9","(s^-10)","Ninth time deriv baryctr rotn frequency (s^-10)",2,1);
  addParameter(defParam,"DM2","%15.3f","DM2","(cm^-3 pc yr^-2)","Second time deriv of dispersion measure (cm**-3 pc yr^-2)",2,1);
  addParameter(defParam,"DM3","%15.3f","DM3","(cm^-3 pc yr^-3)","Third time deriv of dispersion measure (cm**-3 pc yr^-3)",2,1);
  addParameter(defParam,"DM4","%15.3f","DM4","(cm^-3 pc yr^-4)","Fourth time deriv of dispersion measure (cm**-3 pc yr^-4)",2,1);
  addParameter(defParam,"DM5","%15.3f","DM5","(cm^-3 pc yr^-5)","Fifth time deriv of dispersion measure (cm**-3 pc yr^-5)",2,1);
  addParameter(defParam,"DM6","%15.3f","DM6","(cm^-3 pc yr^-6)","Sixth time deriv of dispersion measure (cm**-3 pc yr^-6)",2,1);
  addParameter(defParam,"DM7","%15.3f","DM7","(cm^-3 pc yr^-7)","Seventh time deriv of dispersion measure (cm**-3 pc yr^-7)",2,1);
  addParameter(defParam,"DM8","%15.3f","DM8","(cm^-3 pc yr^-8)","Eighth time deriv of dispersion measure (cm**-3 pc yr^-8)",2,1);
  addParameter(defParam,"DM9","%15.3f","DM9","(cm^-3 pc yr^-9)","Ninth time deriv of dispersion measure (cm**-3 pc yr^-9)",2,1);
  addParameter(defParam,"OMDOT_2","%10.5f","OMDOT_2","(deg)","periastron advance [2nd orbit] (degrees/yr)",3,1);
  addParameter(defParam,"OMDOT_3","%10.5f","OMDOT_3","(deg)","periastron advance [3nd orbit] (degrees/yr)",3,1);
  addParameter(defParam,"PBDOT_2","%10.5f","PBDOT_2"," ","1st time-deriv of binary period [2nd orbit]",3,1);
  addParameter(defParam,"PBDOT_3","%10.5f","PBDOT_3"," ","1st time-deriv of binary period [3rd orbit]",3,1);
  addParameter(defParam,"PPNGAMMA","%10.5f","PPNGAMMA","( )","PPN parameter gamma",3,1);
  addParameter(defParam,"SINI","%10.5f","SINI","( )","Sine of inclination angle",3,1);
  addParameter(defParam,"SINI_2","%10.5f","SINI_2","( )","Sine of inclination angle [2nd orbit]",3,1);
  addParameter(defParam,"SINI_3","%10.5f","SINI_3","( )","Sine of inclination angle [3rd orbit]",3,1);
  addParameter(defParam,"KOM","%10.5f","KOM","(deg)","Longitude of the ascending node",3,1);
  addParameter(defParam,"KIN","%10.5f","KIN","(deg)","Inclination angle",3,1);
  addParameter(defParam,"MTOT","%10.5f","MTOT","(Msun)","Total system mass (solar masses)",3,0);
  addParameter(defParam,"M2","%10.5f","M2","(Msun)","Companion mass (solar masses)",3,1);
  addParameter(defParam,"M2_2","%10.5f","M2_2","(Msun)","Companion mass [2nd orbit] (solar masses)",3,1);
  addParameter(defParam,"M2_3","%10.5f","M2_3","(Msun)","Companion mass [3rd orbit] (solar masses)",3,1);
  addParameter(defParam,"DTHETA","%10.5f","DTHETA","(10^-6)","Relativistic deformation of the orbit (-6)",3,1);
  addParameter(defParam,"XOMDOT","%10.5f","XOMDOT","(deg/yr)","Rate of periastron advance minus GR prediction (deg/yr)",3,1);
  addParameter(defParam,"DR","%10.5f","DR"," ","Relativistic deformation of the orbit (not fit)",3,1);
  addParameter(defParam,"A0","%10.5f","A0"," ","Aberration parameter A0 (microsec - not fit)",3,1);
  addParameter(defParam,"B0","%10.5f","B0"," ","Aberration parameter B0 (microsec - not fit)",3,1);
  addParameter(defParam,"BP","%10.5f","BP"," ","Tensor multi-scalar parameter beta-prime (not fit)",3,1);
  addParameter(defParam,"BPP","%10.5f","BPP"," ","Tensor multi-scalar parameter beta-prime-prime (not fit)",3,1);
  addParameter(defParam,"STIG","%10.5f","STIG"," ","STIG Parameter for Freire & Wex (2011) model",3,1);
  addParameter(defParam,"H3","%10.5f","H3"," ","H3 Parameter for Freire & Wex (2011) model",3,1);
  addParameter(defParam,"H4","%10.5f","H4"," ","H4 Parameter for Freire & Wex (2011) model",3,1);
  addParameter(defParam,"MASS_Q","%10.5f","MASS_Q"," ","MASS_Q = M_psr/M_comp",3,0);
  addParameter(defParam,"MASS_Q_2","%10.5f","MASS_Q_2"," ","MASS_Q = M_psr/M_comp [second orbit]",3,0);
  addParameter(defParam,"OM_ASC","%10.5f","OM_ASC"," ","OM_ASC",3,0);
  addParameter(defParam,"UPRMASS","%10.5f","UPRMASS"," ","90% confidence upper companion mass limit, i=26deg (M_sun)",3,0);
  addParameter(defParam,"MINOMDOT","%10.4f","MINOMDOT"," ","Minimum Omega dot, assuming sin i=1 and M_ns=1.35Mo (deg/yr)",3,0);
  addParameter(defParam,"TDMP","%5.0f","TDMP","(sec)","Ideal dump time for pulsar  (sec)",7,0);
  addParameter(defParam,"TDMPMIN","%8.0f","TDMPMIN","(sec)","Minimum dump time for pulsar  (sec)",7,0);
  addParameter(defParam,"EPHVER","%5.0f","EPHVER"," ","Ephemeris version",7,1);
  addParameter(defParam,"UNITS","%-8.8s","UNITS"," ","Units definition",7,1);
  addParameter(defParam,"NSPAN","%5.0f","NSPAN"," ","Minutes for polyco span",7,0);
  addParameter(defParam,"NCOEF","%5.0f","NCOEF"," ","Number of coefficients in polyco",7,0);
  addParameter(defParam,"NBIN","%5.0f","NBIN"," ","Number of bins in the profile",7,0);
  addParameter(defParam,"NTCOEF","%5.0f","NTCOEF"," ","Number of time coefficients in prediction",7,0);
  addParameter(defParam,"NFCOEF","%5.0f","NFCOEF"," ","Number of frequency coefficients in prediction",7,0);
  addParameter(defParam,"PREDLEN","%5.0f","PREDLEN"," ","Prediction length (s)",7,0);




  addParameter(defParam,"NSUB","%5.0f","NSUB"," ","Number of subbands to store",7,0);
  addParameter(defParam,"GLEP","%8.2f","GLEP","(MJD)","Glitch epoch (MJD)",12,1);
  addParameter(defParam,"GLPH","%8.2f","GLPH"," ","Glitch phase increment",12,1);
  addParameter(defParam,"GLF0","%8.2f","GLF0","(Hz)","Glitch permanent pulse frequency increment (Hz)",12,1);
  addParameter(defParam,"GLF1","%8.2f","GLF1","(Hz)","Glitch permanent frequency deriv increment (s^-2)",12,1);
  addParameter(defParam,"GLF0D","%8.2f","GLF0D","(Hz)","Glitch pulse frequency increment (Hz)",12,1);
  addParameter(defParam,"GLTD","%8.2f","GLTD","(dayd)","Glitch Decay time constant (d)",12,1);
  addParameter(defParam,"WBNCHN","%7.0f","WBNCHN"," ","WB correlator: number of frequency channels (nearest submultiple of orig NCHAN)",8,0);
  addParameter(defParam,"WBTSUB","%7.3f","WBTSUB"," ","WB correlator: subint time (nearest multiple of cycle time)",8,0);
  addParameter(defParam,"WBNBIN","%7.0f","WBNBIN"," ","WB correlator: nr of bins (nearest submult of original nbin)",8,0);
  addParameter(defParam,"FBNCHN","%7.0f","FBNCHN"," ","Filter bank: number of frequency channels (nearest submultiple of orig NCHAN)",9,0);
  addParameter(defParam,"FBTSUB","%7.3f","FBTSUB"," ","Filter bank: subint time",9,0);
  addParameter(defParam,"FBNBIN","%7.0f","FBNBIN"," ","Filter bank: nr of bins",9,0);
  addParameter(defParam,"CPCOH","%6.0f","CPCOH"," ","CPSR: 1 or 0, whether to coherently dedisperse or not",10,0);
  addParameter(defParam,"CPNCHN","%7.0f","CPNCHN"," ","CPSR: channels during FFTing",10,0);
  addParameter(defParam,"CPNFFT","%7.0f","CPNFFT"," ","CPSR: N pts in FFT",10,0);
  addParameter(defParam,"CPNCHN2","%8.0f","CPNCHN2"," ","CPSR: Final #channels after fscrunching",10,0);
  addParameter(defParam,"CPTSUB","%7.3f","CPTSUB"," ","CPSR: subint time",10,0);
  addParameter(defParam,"CPNBIN","%7.0f","CPNBIN"," ","CPSR: nr of bins",10,0);

  addParameter(defParam,"PAR1","%16.7g","PAR1"," ","User parameter 1",11,0);
  addParameter(defParam,"PAR2","%16.7g","PAR2"," ","User parameter 2",11,0);
  addParameter(defParam,"PAR3","%16.7g","PAR3"," ","User parameter 3",11,0);
  addParameter(defParam,"PAR4","%16.7g","PAR4"," ","User parameter 4",11,0);
}

/* ******************************************* */
/* p1 = parameter label                        */
/* p2 = string used for printing in short form */
/* p3 = top line of header                     */
/* p4 = bottom line of header                  */
/* p5 = help description                       */
/* helpType = grouping used when displaying    */
/*      help information (e.g. groups binary   */
/*      parameters together)                   */
/* ephem = parameter displayed in which        */
/*         ephemeris mode                      */
/* ******************************************* */
void addParameter(paramtype *defParam,char *p1,char *p2,char *p3,char *p4,char *p5,int helpType,int ephem)
{
  static unsigned int count=0;
  strcpy(defParam[count].shortStr,p2);
  if (strstr(p2,"s")==0) 
    defParam[count].numeric=1;
  else
    defParam[count].numeric=0;
  strcpy(defParam[count].header1,p3);
  strcpy(defParam[count].header2,p4);
  strcpy(defParam[count].help,p5);
  defParam[count].helpType=helpType;
  defParam[count].ephem=ephem;
  strcpy(pcat_parameters[count++],p1);
}

int nearInt(double x){
  int i;
  if(x>0.){
    i=(int)(x+0.5);
  }
  else{
    i=(int)(x-0.5);
  }
  return(i);
}

float posAngle(float a1,float b1,float a2,float b2)
{
  float posAngle;
  double da, x, y;

  da = (double)a2 - (double)a1;
  y = sin(da) * cos(b2);
  x = sin(b2) * cos(b1) - cos(b2) * sin(b1) * cos(da);
  posAngle = ( x != 0.0 || y != 0.0 ) ? atan2 ( y, x ) : 0.0;

  return posAngle;
}

/* Conversion between equatorial and ecliptic coordinates */
void convertEcliptic(double raj,double decj,double *elong,double *elat)
{
  double sinb,beta,x,y,lambdap,lambda;
  double deg2rad = M_PI/180.0;
  double epsilon = 23.439292*deg2rad;
  /*  double epsilon = 23.441884*deg2rad;*/

  sinb = sin(decj)*cos(epsilon)-cos(decj)*sin(epsilon)*sin(raj);
  beta = asin(sinb);
  y = sin(raj)*cos(epsilon)+tan(decj)*sin(epsilon);
  x = cos(raj);
  
  lambdap = atan2(y,x);
  if (lambdap<0) lambda=lambdap+2*M_PI;
  else lambda = lambdap;

  *elong = lambda;
  *elat  = beta;
}

void convertGalactic(double raj,double decj,double *gl,double *gb)
{
  double sinb,y,x,at;
  double rx,ry,rz,rx2,ry2,rz2;
  double deg2rad = M_PI/180.0;
  double gpoleRAJ = 192.85*deg2rad;
  double gpoleDECJ = 27.116*deg2rad;
  double rot[4][4];

  /* Note: Galactic coordinates are defined from B1950 system - e.g. must transform from J2000.0 
     equatorial coordinates to IAU 1958 Galactic coords */
  
  /* Convert to rectangular coordinates */
  rx = cos(raj)*cos(decj);
  ry = sin(raj)*cos(decj);
  rz = sin(decj);

  /* Now rotate the coordinate axes to correct for the effects of precession */
  /* These values contain the conversion between J2000 and B1950 and from B1950 to Galactic */
  rot[0][0] = -0.054875539726;
  rot[0][1] = -0.873437108010;
  rot[0][2] = -0.483834985808;
  rot[1][0] =  0.494109453312;
  rot[1][1] = -0.444829589425;
  rot[1][2] =  0.746982251810;
  rot[2][0] = -0.867666135858;
  rot[2][1] = -0.198076386122;
  rot[2][2] =  0.455983795705;

  rx2 = rot[0][0]*rx + rot[0][1]*ry + rot[0][2]*rz;
  ry2 = rot[1][0]*rx + rot[1][1]*ry + rot[1][2]*rz;
  rz2 = rot[2][0]*rx + rot[2][1]*ry + rot[2][2]*rz;

  /* Convert the rectangular coordinates back to spherical coordinates */
  *gb = asin(rz2);
  *gl = atan2(ry2,rx2);
  if (*gl < 0) (*gl)+=2.0*M_PI;
}




