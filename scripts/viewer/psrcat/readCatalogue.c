#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "psrcat.h"
#include <string.h>
#include <ctype.h>
#include "wc_strncmp.h"
#include "evaldefs.h"

void alwaysDerive(pulsar *psr,paramtype *defParam);
int nint_derived(double x);
int checkBoundary(char *boundary,pulsar *psr);
void check_parameter(char* fline,int *haveP,int *haveF,int *haveJ2000,int *haveEclip,int *haveDM,int *haveDistDM,int *haveDistDM1,int *haveBIN,int *havePB,int *haveOM,int *haveECC,int *haveEPS1,int *haveEPS2,char *current_psr,int *bintype,int *errornum);
void check_error(char *fline,char *current_psr,int *errornum);
void check_values(char *fline,int *errornum,char *current_psr);
void check_reference(char* fline,int *errornum,char *current_psr);
void checkCatalogue(FILE *fin,char *fname);

int readCatalogue(pulsar *psr,linkedList *list,int *npsr,char name[MAX_PSR][MAX_NAMELEN],
		  int iname,int *haveName,char *filterStr,int *ptr,char sptr[MAX_PLIST][MAX_STRLEN],int np,
		  char *dbFile,int merge,int willmerge,char *boundary,paramtype *defParam,int checkCat,
		  char alias1[MAX_ALIASES][MAX_STRLEN],char alias2[MAX_ALIASES][MAX_STRLEN],int aliasCount,int exactMatch)
{
  FILE *fin;
  register unsigned int i,j,k,filter;
  char line[2*MAX_STRLEN],dummy[MAX_STRLEN],value[MAX_STRLEN];
  char *strptr,*paramLab,*strval,*path;
  int badval,ival,found=-1;
  int o_npsr,aliasip,ip,ip2;
  int maxpsr = *npsr;
  aliasip = getParam("ALIAS",NULL);

  if (strlen(dbFile)==0)
    {
      if (merge==1)
	{
	  printf("Error: no catalogue file to merge\n");
	  exit(1);
	}
      path = getenv("PSRCAT_FILE");
      if (path == NULL) /* Use the default database */
	strcpy(dbFile,"/pulsar/psr/runtime/psrcat/psrcat.db");
      else
	strcpy(dbFile,path);
    }

  /* *************************** */
  /* Check if file can be opened */
  /* *************************** */
  if (!(fin = fopen(dbFile,"r")))
    {
      printf("PSRCAT: Unable to open catalogue file: %s\n",dbFile);
      return 1;
    }

  if (merge==0)
    {
      for (j=*npsr;j<MAX_PARAM;j++)
	{
	  psr[0].param[j].set1=0;
	  psr[0].param[j].set2=0;
	  psr[0].param[j].set3=0;
	}
    }

  if (checkCat==1)
    {
      checkCatalogue(fin,dbFile);
      /*      printf("-----------------"); */
      return 0;
    }
  /* *************************** */
  /* Start reading the catalogue */
  /* *************************** */

  while (!feof(fin))
    {
      if (fgets(line,2*MAX_STRLEN,fin)==NULL)
	break;

      if (strlen(line)==1 && line[0]=='\n') break;
      paramLab = strtok(line," \t");
      if (line[0]=='@')
	{
	  list[*npsr].current = &psr[*npsr];
	  list[*npsr].post    = &list[(*npsr)+1];
	  /* psr[*npsr].merged=0;*/ /* Maybe a problem with this line? */
	  filter=0;
	  /* Check alias list */

	  for (i=0;i<aliasCount;i++)
	    {
	      if (strcmp(psr[*npsr].param[0].val,alias1[i])==0)
		{
		  psr[*npsr].param[aliasip].set1=1;
		  strcpy(psr[*npsr].param[aliasip].val,alias2[i]);
		}
	    }
	  if (iname>0) /* Check for pulsar name on the command line */
	    {
	      filter=1;
	      for (i=iname;i--;)
		{
		  if (wc_strncmp(name[i],psr[*npsr].param[0].val,-1,0,1)==0 ||  /* Check jname */
		      (psr[*npsr].param[1].set1 == 1 && 
		       wc_strncmp(name[i],psr[*npsr].param[1].val,-1,0,1)==0)
		      || (psr[*npsr].param[aliasip].set1 == 1 &&
			  wc_strncmp(name[i],psr[*npsr].param[aliasip].val,-1,0,1)==0))
		    /* Check bname and alias */
		    {
		      haveName[i] = 1;
		      filter=0;
		      break;
		    }
		}
	    }

	  if (filter==0)
	    alwaysDerive(&psr[*npsr],defParam);
	  if (merge!=1 && filter==0 && strlen(boundary)>0) /* Have boundary for position */
	    filter = checkBoundary(boundary,&psr[*npsr]);

 	  if (willmerge==0 && strlen(filterStr)>9 && filter==0) /* logical filter expression */
	    {
	      if (formExpression(filterStr,&psr[*npsr],&badval,exactMatch)==1)
		filter=0;
	      else
		filter=1;
	    }

	  if (filter==0)
	    {
	      /* Check that we have all parameters */
	      if (willmerge==0)
		{
		  for (j=np;j--;)
		    {
		      if (psr[*npsr].param[ptr[j]].set1 != 1)
			deriveParameter(sptr[j],&psr[*npsr],ptr[j]);
		    } 
		}
	      if (found==-1)
		{		  
		  (*npsr)++;
		  maxpsr++;
		}
	    }
	  if (found==-1)
	    {
	      strcpy(psr[*npsr].param[1].val,"");
	      for (j=MAX_PARAM;j--;)
		{
		  psr[*npsr].param[j].set1=0;
		  psr[*npsr].param[j].set2=0;
		  psr[*npsr].param[j].set3=0;
		}
	    }
	  else
	    {
	      found=-1;
	      *npsr = o_npsr;
	    }
	}
      else
	{
	  /* ****************** */
	  /* Now read the value */
	  /* ****************** */
	  strval = strtok(NULL," \t\n");
	  if (strval==NULL) {
	    //	    	    printf("In here with >%s<\n",paramLab);
	    if (strstr(paramLab,"PSRCAT_CLEAR")!=NULL)
	      {
		//		printf("Got into this bit: %d\n",MAX_PARAM);
		// Starting at 3 as 0, 1 and 2 are the pulsar names
		for (j=3;j<MAX_PARAM;j++)
		  {
		    //		    printf("%d In here with %d %g %d\n",*npsr,j,psr[*npsr].param[j].shortVal,psr[*npsr].param[j].set1);
		    //		    if (psr[*npsr].param[j].set1==1) printf("Switching off %s\n",psr[*npsr].param[j].val);
		    psr[*npsr].param[j].set1=0;
		    psr[*npsr].param[j].set2=0;
		    psr[*npsr].param[j].set3=0;
		  }
	      }
	  }
	  else
	    {
	      if ((ival = getParam(paramLab,NULL))==-1)
		{
		    if (verbose==1) printf("Warning parameter in catalogue: %s\n",paramLab);
		    if (strcmp(paramLab,"E")==0)
		      {
			printf("ERROR: Have 'E' in catalogue %s instead of ECC \n",dbFile);
			printf("... must fix before rerunning psrcat\n");
			exit(1);
		      }
		}
	      else
		{
		  /* **************************************************************************** */
		  /* If merging catalogues then check if this pulsar should be merged or inserted */
		  /* **************************************************************************** */
		  if (found==-1 && merge==1 && (strcmp(paramLab,"PSRJ")==0 || strcmp(paramLab,"PSRB")==0))
		    {
		      found=-1;
		      /*		  printf("Here with %d >%s< >%s<\n",maxpsr,); */
		      for (i=maxpsr;i--;)
			{
			  if (strcmp(psr[i].param[0].val,strval)==0 || strcmp(psr[i].param[1].val,strval)==0)
			    {
			      found=i;
			      break;
			    }
			  if (strcmp(psr[i].param[aliasip].val,strval)==0)
			    {
			      found=i;
			      strcpy(strval,psr[i].param[0].val);
			      break;
			    }
			  
			}
		      if (found!=-1)
			{
			  o_npsr = *npsr;
			  *npsr = found;
			}
		      psr[*npsr].merged=1;

		    }	         
		  if (strcmp(paramLab,"DECJ")==0 || strcmp(paramLab,"RAJ")==0)  /* Check have XX:YY:ZZ.zzz */
		    {
		      int ip_elat,ip_elong;
		      /* Remove elat and elong if already read into record */
		      ip_elong = getParam("ELONG",NULL);
		      ip_elat  = getParam("ELAT",NULL);
		      if (psr[*npsr].param[ip_elong].set1==1) psr[*npsr].param[ip_elong].set1=0;
		      if (psr[*npsr].param[ip_elat].set1==1) psr[*npsr].param[ip_elat].set1=0;
		      
		      for (k=strlen(strval);k>3;k--)
			{
			  if (strval[k]=='.' && strval[k-2]==':')
			    {
			      strcpy(dummy,strval+k-1);
			      strcpy(value,strval);
			      value[k-1]='\0';
			      strcat(value,"0");
			      strcat(value,dummy);
			      strval = value;
			      break;
			    }
			  else if (strval[k]==':' && k+2==strlen(strval))
			    {
			      strcpy(dummy,strval+k+1);
			      strcpy(value,strval);
			      value[k+1]='\0';
			      strcat(value,"0");
			      strcat(value,dummy);
			      strval = value;
			      break;
			    }
			}
		    }
		  else if (strcmp(paramLab,"ELAT")==0 || strcmp(paramLab,"ELONG")==0)
		    {
		      int ip_raj,ip_decj;
		      ip_raj  = getParam("RAJ",NULL);
		      ip_decj = getParam("DECJ",NULL);
		      if (psr[*npsr].param[ip_raj].set1==1) psr[*npsr].param[ip_raj].set1=0;
		      if (psr[*npsr].param[ip_decj].set1==1) psr[*npsr].param[ip_decj].set1=0;
		    }
		  else if (strcmp(paramLab,"P0")==0)
		    {
		      int ip;
		      ip = getParam("F0",NULL);
		      if (psr[*npsr].param[ip].set1==1) psr[*npsr].param[ip].set1=0;
		    }
		  else if (strcmp(paramLab,"P1")==0)
		    {
		      int ip;
		      ip = getParam("F1",NULL);
		      if (psr[*npsr].param[ip].set1==1) psr[*npsr].param[ip].set1=0;
		    }
		  else if (strcmp(paramLab,"F0")==0)
		    {
		      int ip;
		      ip = getParam("P0",NULL);
		      if (psr[*npsr].param[ip].set1==1) psr[*npsr].param[ip].set1=0;
		    }
		  else if (strcmp(paramLab,"F1")==0)
		    {
		      int ip;
		      ip = getParam("P1",NULL);
		      if (psr[*npsr].param[ip].set1==1) psr[*npsr].param[ip].set1=0;
		    }
		  
		  if (strcmp(paramLab,"DECJ")==0 && strval[0]!='+' && strval[0]!='-')
		    sprintf(psr[*npsr].param[ival].val,"+%s",strval);
		  else
		    strcpy(psr[*npsr].param[ival].val,strval);
		  sscanf(strval,"%lf",&(psr[*npsr].param[ival].shortVal));
		  if (psr[*npsr].merged==1 && 
		      (strcmp(paramLab,"BINARY")==0))
		    {
		      /* Clear all standard binary model parameters */
		      ip  = getParam("A1",NULL); psr[*npsr].param[ip].set1 = 0; psr[*npsr].param[ip].shortVal=0.0;
		      ip  = getParam("PB",NULL); psr[*npsr].param[ip].set1 = 0; psr[*npsr].param[ip].shortVal=0.0;
		      ip  = getParam("ECC",NULL); psr[*npsr].param[ip].set1 = 0;psr[*npsr].param[ip].shortVal=0.0;
		      ip  = getParam("T0",NULL); psr[*npsr].param[ip].set1 = 0;psr[*npsr].param[ip].shortVal=0.0;
		      ip  = getParam("OM",NULL); psr[*npsr].param[ip].set1 = 0;psr[*npsr].param[ip].shortVal=0.0;
		      ip  = getParam("TASC",NULL); psr[*npsr].param[ip].set1 = 0;psr[*npsr].param[ip].shortVal=0.0;
		      ip  = getParam("EPS1",NULL); psr[*npsr].param[ip].set1 = 0;psr[*npsr].param[ip].shortVal=0.0;
		      ip  = getParam("EPS2",NULL); psr[*npsr].param[ip].set1 = 0;psr[*npsr].param[ip].shortVal=0.0;
		    }
		  if (psr[*npsr].merged==1 && strcmp(paramLab,"EPHVER")==0)
		    {
		      int units = getParam("UNITS",NULL);
		      if (psr[*npsr].param[ival].shortVal==5)
			{
			  psr[*npsr].param[units].shortVal = 1;
			  strcpy(psr[*npsr].param[units].val,"TCB");
			  psr[*npsr].param[units].set1 = 1;
			}
		      else
			{
			  psr[*npsr].param[units].set1 = 1;
			  psr[*npsr].param[units].shortVal = -1;
			  strcpy(psr[*npsr].param[units].val,"TDB");
			}
		    }
		  if (psr[*npsr].merged==1 && strcmp(paramLab,"UNITS")==0)
		    {
		      int eph = getParam("EPHVER",NULL);
		      if (strcmp(psr[*npsr].param[ival].val,"TCB")==0)
			{
			  psr[*npsr].param[eph].shortVal = 5;
			  strcpy(psr[*npsr].param[eph].val,"5");
			  psr[*npsr].param[eph].set1 = 1;
			}
		      else
			{
			  psr[*npsr].param[eph].set1 = 1;
			  psr[*npsr].param[eph].shortVal = 2;
			  strcpy(psr[*npsr].param[eph].val,"2");
			}
		    }
		  if (psr[*npsr].merged==1 && 
		      (strcmp(paramLab,"PMELAT")==0 || strcmp(paramLab,"PMELONG")==0))
		    {
		      ip  = getParam("PMRA",NULL);
		      ip2 = getParam("PMDEC",NULL);
		      if (psr[*npsr].param[ip].set1 == 1 || psr[*npsr].param[ip2].set1) 
			{
			  psr[*npsr].param[ip].shortVal=0.0;
			  psr[*npsr].param[ip2].shortVal=0.0;
			  psr[*npsr].param[ip].set1=0;
			  psr[*npsr].param[ip2].set1=0;
			}
		    }
		  if (psr[*npsr].merged==1 && 
		      (strcmp(paramLab,"PMRA")==0 || strcmp(paramLab,"PMDEC")==0))
		    {
		      ip  = getParam("PMELAT",NULL);
		      ip2 = getParam("PMELONG",NULL);
		      if (psr[*npsr].param[ip].set1 == 1 || psr[*npsr].param[ip2].set1) 
			{
			  psr[*npsr].param[ip].shortVal=0.0;
			  psr[*npsr].param[ip2].shortVal=0.0;
			  psr[*npsr].param[ip].set1=0;
			  psr[*npsr].param[ip2].set1=0;
			}
		    }
		  psr[*npsr].param[ival].set1 = 1;
		  /* Check for size required to print value */
		  if (pcat_maxSize[ival] < strlen(strval)) pcat_maxSize[ival]=strlen(strval);
		  /*	      if (pcat_descend==4) exit(1); */
		  /* The next column could be non-existent, could be error or ref */
		  strptr = strtok(NULL," \t\n");
		  if (strptr!=NULL)  /* Have something */
		    {
		      if (isdigit((int)strptr[0])) /* Numeric, therefore uncertainty */
			{
			  strcpy(psr[*npsr].param[ival].err,strptr);
			  /* Form expanded version of the uncertainty */
			  psr[*npsr].param[ival].error_expand = 
			    errscale(strval,psr[*npsr].param[ival].err);
			  psr[*npsr].param[ival].set2 = 1;
			  pcat_errZero[ival]=-1;
			  /* Probably a reference as well ? */		      
			  
			  if ((strptr = strtok(NULL," \t\n")) != NULL)
			    {
			      strcpy(psr[*npsr].param[ival].ref,strptr);
			      psr[*npsr].param[ival].set3 = 1;
			      pcat_refZero[ival]=-1;
			    }
			}		  
		      else
			{
			  strcpy(psr[*npsr].param[ival].ref,strptr);
			  psr[*npsr].param[ival].set3 = 1;
			  pcat_refZero[ival]=-1;
			}
		    }
		}
	      /* Check whether the parameter = NULL */
	      if (strcasecmp(psr[*npsr].param[ival].val,"NULL")==0)
		{
		  psr[*npsr].param[ival].set1 = 0;
		  psr[*npsr].param[ival].set2 = 0;
		  psr[*npsr].param[ival].set3 = 0;
		}
	    }
	}
    }
  fclose(fin);
  return 0;
}



/* Paul Harrison */
double errscale(char * valstr, char * errstr)
{
  double temp=1;
  int i=0;
  char ovalstr[MAX_STRLEN];

				/* test for the cases where the error */
				/* represents the absolute value i.e. */
				/* if negative or if there is a */
				/* decimal point in string */

  if (errstr[0] == '-' || strchr(errstr,'.')!=NULL) return 1.0;
  strcpy(ovalstr,valstr);
  if ((valstr=strchr(ovalstr,'.'))==NULL)
    {
      if (strchr(ovalstr,'E')!=NULL)
	valstr=ovalstr;
      else if (strchr(ovalstr,'e')!=NULL)
	valstr=ovalstr;
    }
  if (valstr!=NULL)
    {
      while (isdigit((int)valstr[++i]));
      if(valstr[i] == 'E' || valstr[i] == 'e')
	{
	  /* redwards 24 Mar 98 added 'e' case below since it was absent!*/
	  if (valstr[i]=='E')
	    sscanf(valstr+i,"E%lf",&temp);
	  else
	    sscanf(valstr+i,"e%lf",&temp);
	  temp = pow(10.0,temp);
	}
      else
	temp = 1;
      temp *= pow(10.0,(double)(-(--i)));
      if ( temp>1.0 && (i=strlen(errstr))>1 && errstr[0] != '-') temp *=
	pow(10.0,(double)(--i));
      return temp;
    }
  else
    {
      return 1;
    }
}

/* badVal = 1 if problem in expression */
double formExpression(char *filterStr,pulsar *psr,int *badval,int exactMatch)
{
  int ip,i,errorFlag;
  char newstr[MAX_FILTERSTR]="";
  char storeStr[MAX_FILTERSTR];
  char *tempptr,storeCharstr[MAX_FILTERSTR];
  char tempStr[MAX_FILTERSTR];

  /*  printf("Starting formExpression\n"); */

  strcpy(storeStr,filterStr);
  /* Search for first occurrence of special characters */
  *badval = 0;
  do {   
    tempptr = strpbrk(storeStr," \t\n()><|!+-*/%&=^~;,");
    if (tempptr!=NULL) 
      {
	strcpy(storeCharstr,tempptr);
	storeCharstr[1] = '\0';
	tempptr[0] = '\0';
      }
    /* Check storeStr for specific functions */
    /*   printf("HAVE STORESTR = :%s:\n",storeStr); */
    if (strcmp(storeStr,"EXIST")==0)
      {
	char *tempptr2;
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	tempptr2 = strpbrk(storeStr,")");
	tempptr2[0] = '\0';
	if ((ip = getParam(storeStr,psr))==-1)
	  {
	    *badval=1;
	    return 0; 
	  }
	if (psr->param[ip].set1==1)
	  strcat(newstr,"1");
	else
	  strcat(newstr,"0");
	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';
      }
    else if (strcmp(storeStr,"SURVEY")==0)
      {
	char *tempptr2;
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	tempptr2 = strpbrk(storeStr,")");
	if (!tempptr2)
	  {
	    printf("Problem in understanding SURVEY command.  Should use e.g. survey(pksmb)\n");
	    *badval=1;
	    exit(1);
	  }
	tempptr2[0] = '\0';
	if ((ip = getParam("SURVEY",psr))==-1)
	  {
	    *badval=1;
	    return 0;
	  }
	if (psr->param[ip].set1!=1)
	  strcat(newstr,"0");
	else
	  {
	    char survLower[1024];

	    /* Surveys are in lower case ! */
	    for (i=strlen(storeStr);i--;)
	      storeStr[i] = tolower(storeStr[i]);
	    for (i=strlen(psr->param[ip].val);i--;)
	      survLower[i] = tolower(psr->param[ip].val[i]);
	    survLower[strlen(psr->param[ip].val)]='\0';
	    //	    printf("Comparing %s %s %s %s\n",psr->param[0].val,psr->param[ip].val,survLower,storeStr);
	    //	    if (strstr(psr->param[ip].val,storeStr)!=NULL)
	    if (exactMatch==0)
	      {
		if (strstr(survLower,storeStr)!=NULL)
		  strcat(newstr,"1");
		else
		  strcat(newstr,"0");
	      }
	    else
	      {
		int kk;
		char *t_tok;
		int found=0;

		t_tok = strtok(survLower,",");
		while (t_tok != NULL)
		  {
		    if (strcmp(t_tok,storeStr)==0)
		      {
			found=1;
			break;
		      }
		    t_tok = strtok(NULL,",");

		  } 

		if (found==1)
		  strcat(newstr,"1"); 
		else
		  strcat(newstr,"0");
	      }
	  }
	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';
      }
    else if (strcmp(storeStr,"REF")==0)
      {
	char *tempptr2,*tempptr3;
	char store[1000],output[1000];

	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);

	tempptr2 = strpbrk(storeStr,",");

	tempptr2[0] = '\0';
	strcpy(store,storeStr);


	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr, tempStr);
	tempptr3 = strpbrk(storeStr,")");
	tempptr3[0] = '\0';

	if ((ip = getParam(store,psr))==-1)
	  {
	    *badval=1;
	    return 0;
	  }
	if (psr->param[ip].set1!=1)
	  strcat(newstr,"0");
	else
	  {
	    strcpy(output,psr->param[ip].ref);
	    upperCase(output);	    
	    if (strstr(output,storeStr)!=NULL)
	      strcat(newstr,"1");
	    else
	      strcat(newstr,"0");
	  }
	strcpy(tempStr,tempptr3+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';
      }
    else if (strcmp(storeStr,"ASSOC")==0)
      {
	char *tempptr2;
	char output[1000];
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	tempptr2 = strpbrk(storeStr,")");
	tempptr2[0] = '\0';
	if ((ip = getParam("ASSOC",psr))==-1)
	  {
	    *badval=1;
	    return 0;
	  }
	if (psr->param[ip].set1!=1)
	  strcat(newstr,"0");
	else
	  {
	    strcpy(output,psr->param[ip].val);
	    upperCase(output);
	    upperCase(storeStr);
	    if (exactMatch==0)
	      {
		if (strstr(output,storeStr)!=NULL)
		  strcat(newstr,"1");
		else
		  strcat(newstr,"0");
	      }
	    else
	      {
		int kk;
		char *t_tok;
		int found=0;

		t_tok = strtok(output,",");
		while (t_tok != NULL)
		  {
		    if (strcmp(t_tok,storeStr)==0)
		      {
			found=1;
			break;
		      }
		    t_tok = strtok(NULL,",");
		  } 

		if (found==1)
		  strcat(newstr,"1"); 
		else
		  strcat(newstr,"0");
	      }

	  }
	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';
      }
    else if (strcmp(storeStr,"BINCOMP")==0)
      {
	char *tempptr2;
	char output[1000];
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	tempptr2 = strpbrk(storeStr,")");
	tempptr2[0] = '\0';
	if ((ip = getParam("BINCOMP",psr))==-1)
	  {
	    *badval=1;
	    return 0;
	  }
	if (psr->param[ip].set1!=1)
	  strcat(newstr,"0");
	else
	  {
	    char *checkBinComp;
	    strcpy(output,psr->param[ip].val);
	    upperCase(output);
	    checkBinComp = strtok(output,"[\n");
	    upperCase(storeStr);
	    if (strstr(checkBinComp,storeStr)!=NULL)
	      strcat(newstr,"1");
	    else
	      strcat(newstr,"0");
	  }
	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';
      }
    else if (strcmp(storeStr,"DISCOVERY")==0)
      {
	char *tempptr2;
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	
	tempptr2 = strpbrk(storeStr,")");
	tempptr2[0] = '\0';
	if ((ip = getParam("SURVEY",psr))==-1)
	  {
	    *badval=1;
	    return 0;
	  }
	if (psr->param[ip].set1!=1)
	  strcat(newstr,"0");
	else
	  {
	    char *first;
	    char value[1024];

	    strcpy(value,psr->param[ip].val);
	    // Extract first survey
	    first = strtok(value," ;,\n");

	    if (exactMatch==0)
	      {
	    //
	    //	    if (strcasestr(first,storeStr)!=0)
	    //	      strcat(newstr,"1");
	    //	    else
	    //	      strcat(newstr,"0");

		if (strcasestr(first,storeStr)!=0)
		  strcat(newstr,"1");
		else
		  strcat(newstr,"0");
	      }
	    else
	      {
		int kk;
		char *t_tok;
		int found=0;

		if (strcasecmp(first,storeStr)==0)
		  strcat(newstr,"1"); 
		else
		  strcat(newstr,"0");
	      }
	  }
	strcpy(tempStr,tempptr2+1);	
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';	
      }
    else if (strcmp(storeStr,"TYPE")==0)
      {
	char *tempptr2;
	int ipb,ip2;
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	tempptr2 = strpbrk(storeStr,")");
	tempptr2[0] = '\0';

	ipb = getParam("BINARY",NULL);
	ip2 = getParam("PB",NULL);
	ip = getParam("TYPE",NULL);
	
	if ((ip = getParam("TYPE",NULL))==-1)
	  strcat(newstr,"0");
	else if (psr->param[ip].set1==1 && strstr(psr->param[ip].val,storeStr)!=NULL)
	  strcat(newstr,"1");
	else if (strcmp(storeStr,"RADIO")==0 && (psr->param[ip].set1==0 || (psr->param[ip].set1==1 &&
									    strstr(psr->param[ip].val,"NRAD")==NULL)))
	  strcat(newstr,"1");
	else if (strcmp(storeStr,"BINARY")==0 && (psr->param[ipb].set1==1 || psr->param[ip2].set1==1))
	  strcat(newstr,"1");
	else
	  strcat(newstr,"0");
	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';	
      }
    else if (strcmp(storeStr,"DMS")==0 || strcmp(storeStr,"D")==0)
      {
	char *tempptr2;
	char store[1000];
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	tempptr2 = strpbrk(storeStr,")");
	tempptr2[0] = '\0';
	sprintf(store,"%.5f",turn_deg(dms_turn(storeStr)));
	strcat(newstr,store);
	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';
      }
    else if (strcmp(storeStr,"HMS")==0 || strcmp(storeStr,"H")==0)
      {
	char *tempptr2;
	char store[1000];
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	tempptr2 = strpbrk(storeStr,")");
	tempptr2[0] = '\0';
	sprintf(store,"%.5f",turn_deg(hms_turn(storeStr)));
	strcat(newstr,store);
	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';
      }
    else if (strcmp(storeStr,"ERROR")==0)
      {
	char *tempptr2;
	strcpy(tempStr,tempptr+1);
	strcpy(storeStr, tempStr);
	tempptr2 = strpbrk(storeStr,")");
	tempptr2[0] = '\0';
	if ((ip = getParam(storeStr,psr))==-1)
	  {
	    *badval=1;
	    return 0;
	  }
	if (psr->param[ip].set1==1 && psr->param[ip].set2==1)
	  {
	    int ierr;
	    char output[100];

	    sscanf(psr->param[ip].err,"%d",&ierr);
	    sprintf(output,"%.6g\0",psr->param[ip].error_expand*ierr); /* Check errors to 6 decimal places */ 
	    strcat(newstr,output);
	  }
	else
	  {
	    *badval=1;
	    return 0;
	  }
	strcpy(tempStr,tempptr2+1);
	strcpy(storeStr,tempStr);
	storeCharstr[0]='\0';
      }
    else if (strlen(storeStr)>0 && (ip = getParam(storeStr,psr))!=-1)
      {
	if (strcmp(storeStr,"RAJ")==0)
	  ip = getParam("RAJD",psr);
	else if (strcmp(storeStr,"DECJ")==0)
	  ip = getParam("DECJD",psr);
	if (psr->param[ip].set1!=1) /* This was commented out -- why??? */
	  {
	    *badval=1;
	    return 0; 
	  }
	strcat(newstr,psr->param[ip].val);	    
	if (tempptr!=NULL)
	  {
	    strcpy(tempStr,tempptr+1);
	    strcpy(storeStr, tempStr);
	  }
      }
    else
      {
	strcat(newstr,storeStr);
	if (tempptr!=NULL) {
	  strcpy(tempStr, tempptr+1);
	  strcpy(storeStr, tempStr);
	}
      }
    if (tempptr!=NULL) strcat(newstr,storeCharstr);
  } while (tempptr!=NULL);
  //    printf("Final string = :%s:\n",newstr);  

  /* Convert to lowercase */
  for (i=strlen(newstr);i--;)
    newstr[i]=tolower(newstr[i]);
  errorFlag = evaluateExpression(newstr);
  //      printf("%s Evaluating ~%s~ output = %f %d\n",psr->param[0].val,newstr,variable[0].value,errorFlag);   

  if (errorFlag)
    {
      *badval = 1;
      return 0;
    }
  return variable[0].value;
  
}

/* Parameters that are always derived: P0, P1, F0 and F1 */
/* also PB, PBDOT, FB0, FB1                              */
void alwaysDerive(pulsar *psr,paramtype *defparam)
{
  int f0,f1,p0,p1;
  int pb,fb0,pbdot,fb1;
  double f0val,p0val,p1val,f1val;
  double fb0val,pbval,pbdotval,fb1val;
  int ierr_f0,ierr_f1,ierr_p0,ierr_p1;
  int ierr_fb0,ierr_fb1,ierr_pb,ierr_pbdot;
  int lv,le,eph,units;
  char msg[MAX_STRLEN];

  /* Copy reference from BNAME to JNAME */
  if (psr->param[1].set3==1)
    {strcpy(psr->param[0].ref,psr->param[1].ref); psr->param[0].set3=1;}
 
  f0 = getParam("F0",NULL);
  f1 = getParam("F1",NULL);
  p0 = getParam("P0",NULL);
  p1 = getParam("P1",NULL);
  pb = getParam("PB",NULL);
  fb0 = getParam("FB0",NULL);
  pbdot = getParam("PBDOT",NULL);
  fb1 = getParam("FB1",NULL);
  eph = getParam("EPHVER",NULL);
  units = getParam("UNITS",NULL);

  /* Derive P0 from F0 */
  if (psr->param[f0].set1 == 1 && psr->param[p0].set1 != 1)
    {
      p0val = psr->param[p0].shortVal = 1.0/psr->param[f0].shortVal;
      if (psr->param[f0].set2==1)
	{
	  sscanf(psr->param[f0].err,"%d",&ierr_f0);
	  psr->param[p0].error_expand = psr->param[f0].error_expand*ierr_f0*p0val*p0val;
	  
	  rnd8(p0val,psr->param[p0].error_expand,1,psr->param[p0].val,&lv,psr->param[p0].err,&le,msg);
	  psr->param[p0].error_expand = errscale(psr->param[p0].val,psr->param[p0].err);
	  psr->param[p0].set2 = 1;
	  psr->param[p0].derived = 1;
	}
      else /* No error */
	{
	  psr->param[p0].set2 = 0;
	  psr->param[p0].error_expand = 0;
	  sprintf(psr->param[p0].val,defparam[p0].shortStr,p0val);
	  psr->param[p0].derived = 1;
	}
      if (psr->param[f0].set3 == 1)
	{
	  strcpy(psr->param[p0].ref,psr->param[f0].ref);
	  psr->param[p0].set3 = 1;
	  psr->param[p0].derived = 1;
	}
      else
	psr->param[p0].set3 = 0;
      psr->param[p0].set1 = 1;  

    }
  else if (psr->param[p0].set1 == 1 && psr->param[f0].set1 != 1) /* Derive frequency from period */
    {
      f0val = psr->param[f0].shortVal = 1.0/psr->param[p0].shortVal;
      p0val = psr->param[p0].shortVal;
      if (psr->param[p0].set2==1)
	{
	  sscanf(psr->param[p0].err,"%d",&ierr_p0);
	  psr->param[f0].error_expand = psr->param[p0].error_expand*ierr_p0*f0val*f0val;
	  rnd8(f0val,psr->param[f0].error_expand,1,psr->param[f0].val,&lv,psr->param[f0].err,&le,msg);
	  psr->param[f0].error_expand = errscale(psr->param[f0].val,psr->param[f0].err);
	  strcpy(psr->param[f0].ref,psr->param[p0].ref);
	  psr->param[f0].set1 = 1; psr->param[f0].set2 = 1; psr->param[f0].set3 = 1;
	  psr->param[f0].derived = 1;
	}
      else /* No error */
	{
	  psr->param[f0].error_expand = 0;
	  sprintf(psr->param[f0].val,defparam[f0].shortStr,f0val);
	  if (psr->param[p0].set3 == 1)
	    {
	      strcpy(psr->param[f0].ref,psr->param[p0].ref);
	      psr->param[f0].set3 = 1;
	      psr->param[f0].derived = 1;
	    }
	  else
	    psr->param[f0].set3 = 0;
	  psr->param[f0].set1 = 1; psr->param[f0].set2 = 0; 
	  psr->param[f0].derived = 1;
	}
      /* Derive F1 from P1 if available */
      if (psr->param[p1].set1 == 1)
	{
	  p1val = psr->param[p1].shortVal;
	  f1val = psr->param[f1].shortVal = -1.0*f0val*f0val*p1val;
	  if (psr->param[p0].set2==1 && psr->param[p1].set2==1)
	    {
	      sscanf(psr->param[p1].err,"%d",&ierr_p1);
	      psr->param[f1].error_expand = sqrt(pow(f0val*f0val*ierr_p1*psr->param[p1].error_expand,2)+
						 pow(2.0*f0val*f0val*f0val*p1val*psr->param[p0].error_expand*
						     ierr_p0,2));
	      rnd8(f1val,psr->param[f1].error_expand,1,psr->param[f1].val,&lv,psr->param[f1].err,&le,msg);
	      psr->param[f1].error_expand = errscale(psr->param[f1].val,psr->param[f1].err);
	      strcpy(psr->param[f1].ref,psr->param[p0].ref);
	      psr->param[f1].set1 = 1; psr->param[f1].set2 = 1; psr->param[f1].set3 = 1;
	      psr->param[f1].derived = 1;
	    }
	  else
	    {
	      strcpy(psr->param[f1].ref,psr->param[p0].ref);
	      sprintf(psr->param[f1].val,defparam[f1].shortStr,f1val);
	      psr->param[f1].set1 = 1; psr->param[f1].set2 = 0; psr->param[f1].set3 = 1;
	      psr->param[f1].derived = 1;
	    }
	}      
      /* Derive P1 from F1 if available */
      if (psr->param[f1].set1 == 1 && psr->param[p1].set1 == 0)
	{
	  f1val = psr->param[f1].shortVal;
	  p1val = psr->param[p1].shortVal = -1.0*p0val*p0val*f1val;
	  if (psr->param[f1].set2==1)
	    {
	      sscanf(psr->param[f1].err,"%d",&ierr_f1);
	      psr->param[p1].error_expand = sqrt(pow(p0val*p0val*ierr_f1*psr->param[f1].error_expand,2)+
						 pow(2.0*p0val*p0val*p0val*f1val*psr->param[f0].error_expand*
						     ierr_f0,2));
	      rnd8(p1val,psr->param[p1].error_expand,1,psr->param[p1].val,&lv,psr->param[p1].err,&le,msg);
	      psr->param[p1].error_expand = errscale(psr->param[p1].val,psr->param[p1].err);
	      psr->param[p1].set2 = 1;
	      psr->param[p1].derived = 1;
	    }
	  else
	    {
	      psr->param[p1].error_expand = 0;
	      sprintf(psr->param[p1].val,defparam[p1].shortStr,p0val);
	      psr->param[p1].set2 = 0;
	  psr->param[p1].derived = 1;
	    }
	  strcpy(psr->param[p1].ref,psr->param[f1].ref);
	  psr->param[p1].set1 = 1;  psr->param[p1].set3 = 1;
	  psr->param[p1].derived = 1;
	}
    }
  /* Check if P1 is set  */
  /* Derive P1 from F1 if available */
  if (psr->param[p1].set1 == 0 && psr->param[f1].set1 == 1)
    {
      p0val = psr->param[p0].shortVal = 1.0/psr->param[f0].shortVal;
      f1val = psr->param[f1].shortVal;
      p1val = psr->param[p1].shortVal = -1.0*p0val*p0val*f1val;
      if (psr->param[f1].set2==1)
	{
	  sscanf(psr->param[f1].err,"%d",&ierr_f1);
	  psr->param[p1].error_expand = sqrt(pow(p0val*p0val*ierr_f1*psr->param[f1].error_expand,2)+
					     pow(2.0*p0val*p0val*p0val*f1val*psr->param[f0].error_expand*
						 ierr_f0,2));
	  rnd8(p1val,psr->param[p1].error_expand,1,psr->param[p1].val,&lv,psr->param[p1].err,&le,msg);
	  psr->param[p1].error_expand = errscale(psr->param[p1].val,psr->param[p1].err);
	  psr->param[p1].set2 = 1;
	  psr->param[p1].derived = 1;
	}
      else
	{
	  psr->param[p1].error_expand = 0;
	  psr->param[p1].set2 = 0;
	  sprintf(psr->param[p1].val,defparam[p1].shortStr,p1val);
	  psr->param[p1].derived = 1;
	}
      psr->param[p1].set1 = 1;  
      if (psr->param[f0].set3==1)
	{
	  strcpy(psr->param[p1].ref,psr->param[f0].ref);
	  psr->param[p1].set3 = 1;
	  psr->param[p1].derived = 1;
	}
      else
	psr->param[p1].set3=0;
    }
  //
  // Orbital period and its derivative
  //
  /* Derive PB from FB0 */
  if (psr->param[fb0].set1 == 1 && psr->param[pb].set1 != 1)
    {
      
      pbval = (1.0/psr->param[fb0].shortVal)/86400.0;
      psr->param[pb].shortVal = pbval;
      pcat_errZero[pb]=-1;
      pcat_refZero[pb]=-1;
	  
      if (psr->param[fb0].set2==1)
	{
	  sscanf(psr->param[fb0].err,"%d",&ierr_fb0);
	  //	  psr->param[pb].error_expand = psr->param[fb0].error_expand/86400.0*ierr_fb0*pbval*pbval;
	  psr->param[pb].error_expand = psr->param[fb0].error_expand*ierr_fb0*pbval*86400.0*pbval;
	  
	  rnd8(pbval,psr->param[pb].error_expand,1,psr->param[pb].val,&lv,psr->param[pb].err,&le,msg);
	  psr->param[pb].error_expand = errscale(psr->param[pb].val,psr->param[pb].err);
	  psr->param[pb].set2 = 1;
	  psr->param[pb].derived = 1;
	}
      else /* No error */
	{
	  psr->param[pb].set2 = 0;
	  psr->param[pb].error_expand = 0;
	  sprintf(psr->param[pb].val,defparam[pb].shortStr,pbval);
	  psr->param[pb].derived = 1;
	}
      if (psr->param[fb0].set3 == 1)
	{
	  strcpy(psr->param[pb].ref,psr->param[fb0].ref);
	  psr->param[pb].set3 = 1;
	  psr->param[pb].derived = 1;
	}
      else
	psr->param[pb].set3 = 0;
      psr->param[pb].set1 = 1;  
      psr->param[pb].derived = 1;

      /* Derive PBDOT from FB1 if available */
      if (psr->param[fb1].set1 == 1 && psr->param[pbdot].set1 == 0)
	{
	  fb1val = psr->param[fb1].shortVal;
	  pbdotval = psr->param[pbdot].shortVal = -1.0*pbval*pbval*fb1val;
	  pcat_errZero[pbdot]=-1;
	  pcat_refZero[pbdot]=-1;

	  if (psr->param[fb1].set2==1)
	    {
	      sscanf(psr->param[fb1].err,"%d",&ierr_fb1);
	      psr->param[pbdot].error_expand = sqrt(pow(pbval*pbval*ierr_fb1*psr->param[fb1].error_expand,2)+
						 pow(2.0*pbval*pbval*pbval*fb1val*psr->param[fb0].error_expand*
						     ierr_fb0,2));
	      rnd8(pbdotval,psr->param[pbdot].error_expand,1,psr->param[pbdot].val,&lv,psr->param[pbdot].err,&le,msg);
	      psr->param[pbdot].error_expand = errscale(psr->param[pbdot].val,psr->param[pbdot].err);
	      psr->param[pbdot].set2 = 1;
	      psr->param[pbdot].derived = 1;
	    }
	  else
	    {
	      psr->param[pbdot].error_expand = 0;
	      sprintf(psr->param[pbdot].val,defparam[pbdot].shortStr,pbdotval);
	      psr->param[pbdot].set2 = 0;
	      psr->param[pbdot].derived = 1;
	    }
	  strcpy(psr->param[pbdot].ref,psr->param[fb1].ref);
	  psr->param[pbdot].set1 = 1;  psr->param[pbdot].set3 = 1;
	  psr->param[pbdot].derived = 1;
	}


    }
  else if (psr->param[pb].set1 == 1 && psr->param[fb0].set1 != 1) /* Derive frequency from period */
    {
      fb0val = psr->param[fb0].shortVal = 1.0/(psr->param[pb].shortVal*86400.0);
      pbval = psr->param[pb].shortVal*86400.0;
      pcat_errZero[fb0]=-1;
      pcat_refZero[fb0]=-1;

      if (psr->param[pb].set2==1)
	{
	  sscanf(psr->param[pb].err,"%d",&ierr_pb);
	  psr->param[fb0].error_expand = psr->param[pb].error_expand*86400.0*ierr_pb*fb0val*fb0val;
	  rnd8(fb0val,psr->param[fb0].error_expand,1,psr->param[fb0].val,&lv,psr->param[fb0].err,&le,msg);
	  psr->param[fb0].error_expand = errscale(psr->param[fb0].val,psr->param[fb0].err);
	  strcpy(psr->param[fb0].ref,psr->param[pb].ref);
	  psr->param[fb0].set1 = 1; psr->param[fb0].set2 = 1; psr->param[fb0].set3 = 1;
	  if (pcat_maxSize[fb0] < strlen(psr->param[fb0].val)) 
	    pcat_maxSize[fb0]=strlen(psr->param[fb0].val);	      
	  psr->param[fb0].derived = 1;
	}
      else /* No error */
	{
	  psr->param[fb0].error_expand = 0;
	  sprintf(psr->param[fb0].val,defparam[fb0].shortStr,fb0val);
	  if (psr->param[pb].set3 == 1)
	    {
	      strcpy(psr->param[fb0].ref,psr->param[pb].ref);
	      psr->param[fb0].set3 = 1;
	    }
	  else
	    psr->param[fb0].set3 = 0;
	  psr->param[fb0].set1 = 1; psr->param[fb0].set2 = 0; 
	  psr->param[fb0].derived = 1;
	}
      /* Derive FB1 from PBDOT if available */
      if (psr->param[pbdot].set1 == 1)
	{
	  pbdotval = psr->param[pbdot].shortVal;
	  fb1val = psr->param[fb1].shortVal = -1.0*fb0val*fb0val*pbdotval;
	  pcat_errZero[fb1]=-1;
	  pcat_refZero[fb1]=-1;

	  if (psr->param[pb].set2==1 && psr->param[pbdot].set2==1)
	    {
	      sscanf(psr->param[pbdot].err,"%d",&ierr_pbdot);
	      psr->param[fb1].error_expand = sqrt(pow(fb0val*fb0val*ierr_pbdot*psr->param[pbdot].error_expand,2)+
						 pow(2.0*fb0val*fb0val*fb0val*pbdotval*psr->param[pb].error_expand*86400.0*
						     ierr_pb,2));
	      rnd8(fb1val,psr->param[fb1].error_expand,1,psr->param[fb1].val,&lv,psr->param[fb1].err,&le,msg);
	      psr->param[fb1].error_expand = errscale(psr->param[fb1].val,psr->param[fb1].err);
	      strcpy(psr->param[fb1].ref,psr->param[pb].ref);
	      psr->param[fb1].set1 = 1; psr->param[fb1].set2 = 1; psr->param[fb1].set3 = 1;
	      pcat_errZero[fb1]=-1;
	      pcat_refZero[fb1]=-1;
	      psr->param[fb1].derived = 1;
	    }
	  else
	    {
	      strcpy(psr->param[fb1].ref,psr->param[pb].ref);
	      sprintf(psr->param[fb1].val,defparam[fb1].shortStr,fb1val);
	      psr->param[fb1].set1 = 1; psr->param[fb1].set2 = 0; psr->param[fb1].set3 = 1;
	  psr->param[fb1].derived = 1;
	    }
	}
    }
  /* Check if PBDOT is set  */
  /* Derive PBDOT from FB1 if available */
      // GH: 20th May 2017: this code seems to be in twice
      /*  if (psr->param[pbdot].set1 == 0 && psr->param[fb1].set1 == 1)
    {
      pbval = psr->param[pb].shortVal = 1.0/psr->param[fb0].shortVal;
      fb1val = psr->param[fb1].shortVal;
      pbdotval = psr->param[pbdot].shortVal = -1.0*pbval*pbval*fb1val;
      pcat_errZero[pbdot]=-1;
      pcat_refZero[pbdot]=-1;

      if (psr->param[fb1].set2==1)
	{
	  sscanf(psr->param[fb1].err,"%d",&ierr_fb1);
	  psr->param[pbdot].error_expand = sqrt(pow(pbval*pbval*ierr_fb1*psr->param[fb1].error_expand,2)+
					     pow(2.0*pbval*pbval*pbval*fb1val*psr->param[fb0].error_expand*
						 ierr_fb0,2));
	  rnd8(pbdotval,psr->param[pbdot].error_expand,1,psr->param[pbdot].val,&lv,psr->param[pbdot].err,&le,msg);
	  psr->param[pbdot].error_expand = errscale(psr->param[pbdot].val,psr->param[pbdot].err);
	  psr->param[pbdot].set2 = 1;
	  psr->param[pbdot].derived = 1;
	}
      else
	{
	  psr->param[pbdot].error_expand = 0;
	  psr->param[pbdot].set2 = 0;
	  sprintf(psr->param[pbdot].val,defparam[pbdot].shortStr,pbdotval);
	}
      psr->param[pbdot].set1 = 1;  
      psr->param[pbdot].derived = 1;

      if (psr->param[fb0].set3==1)
	{
	  strcpy(psr->param[pbdot].ref,psr->param[fb0].ref);
	  psr->param[pbdot].set3 = 1;
	}
      else
	psr->param[pbdot].set3=0;
	} */
  // End orbital period section


  //  if (psr->param[eph].set1 == 0) /* Use EPHVER 2 as default */
  //  {
  //    sprintf(psr->param[eph].val,"2");
  //    psr->param[eph].shortVal = 2;
  //    psr->param[eph].set1 = 1;
  //  }

  // Choose between EPHVER and UNITS flags
  /*  if (psr->param[eph].set1 == 1 && psr->param[units].set1 == 1)
    {
      // Reset UNITS based on units
      if (strcmp(psr->param[units].val,"TDB")==0){
	  sprintf(psr->param[eph].val,"2");
	  psr->param[eph].shortVal = 2;
	  psr->param[eph].set1 = 1;
      }
      else{
	  sprintf(psr->param[eph].val,"5");
	  psr->param[eph].shortVal = 5;
	  psr->param[eph].set1 = 1;
      }

	
    }
  */
    if (psr->param[units].set1 == 0) // Use UNITS TDB as default, but check for EPHVER 5 
      {
	if (psr->param[eph].set1 == 1)
	  {
	    if (psr->param[eph].shortVal == 2){
	      sprintf(psr->param[units].val,"TDB");
	      psr->param[units].shortVal = -1;
	      psr->param[units].set1 = 1;
	    }
	    else if (psr->param[eph].shortVal == 5) {
	      sprintf(psr->param[units].val,"TCB");
	      psr->param[units].shortVal = 1;
	      psr->param[units].set1 = 1;
	    }
	}
      else
	{
	  sprintf(psr->param[units].val,"TDB");
	  psr->param[units].shortVal = -1;
	  psr->param[units].set1 = 1;
	  sprintf(psr->param[eph].val,"2");
	  psr->param[eph].shortVal = 2;
	  psr->param[eph].set1 = 1;
	}
    }

  //  else
  //    printf("ALREADY SET\n");

}

int rnd8(double rval,double rerr,int ifac,char *cval,int *lv,char *cerr,int *le,char *msg)
{
  double vv, ee, xv, xe;
  int ixv, ixe, iee, j, ivv, ise, irnd, ilim,ret,ret_lv,ret_le;
  char cexp[9], fmt[12];
  char temp[1024];

  ilim = 4;

  strcpy(cval," ");
  strcpy(cerr," ");
  strcpy(msg," ");
  ret = 0;
  ret_lv = 0;
  ret_le = 0;

  /* Get scale factors */
  vv = fabs(rval);
  ee = rerr*ifac;
  if (vv>0.0)
    xv=log10(vv);
  else
    xv=0.0;

  ixv=fabs(xv);
  if (xv<0.0)ixv=-ixv-1;

  xe = log10(ee/2.0)+1.0e-10;
  ixe = fabs(xe);
  if (xe<0.0)ixe=-ixe-1; 
  ise=ixe-ixv; /* Scale of error wrt value */

  ixv = -ixv; /* Reverse signs of scale factors */
  ixe = -ixe;
  ise = -ise;

  /* Check for encoding as integer */
  if (xv>=log10(2.0) && xv<(ilim+10) && ee>=2.0)
    {
      irnd=xe;
      ivv=nint_derived(vv);
      if (irnd>1)
	{
	  irnd=nint_derived(pow(10.0,irnd));
	  ivv = nint_derived(vv/irnd)*irnd;
	}
      if (ivv!=0)
	ret_lv=log10((double)ivv+1.0e-10)+2.0;

      if (rval<0.0) ivv=-ivv;
      sprintf(fmt,"%%%dd",ret_lv);
      sprintf(cval,fmt,ivv);
    }
  else /* Encode as real */
    {
      vv=rval*pow(10.0,ixv); /* Scale for mantissa */
      ee=ee*pow(10.0,ixe);   /* scale error */

      if (ixv<-ilim || ixv>ilim) /* Use exponent notation */
	{
	  if (ise<1) /* If err > 0.2, print val as fn.1 and scale error */
	    {
	      ee=ee*pow(10.0,1-ise);
	      ise=1; 
	    }
	  strcpy(cexp," ");
	  sprintf(cexp,"%d",-ixv); 
	  j=0; /* Strip off leading blanks */
	  do
	    {
	      j=strlen(cexp)-1;
	      if (j==0){
		strcpy(temp,cexp+1);
		strcpy(cexp,temp);
	      }
	    }
	  while (j==0);
	  if (vv<0) /* allow space for - sign */
	    sprintf(fmt,"%%%d.%df",ise+4,ise);
	  else
	    sprintf(fmt,"%%%d.%df",ise+3,ise);
	  sprintf(cval,fmt,vv);
	  if (cval[0]==' ') 
	    {
	      strcpy(temp,cval+1);
	      strcpy(cval,temp);
	    }
	  ret_lv = strlen(cval)-1;
	  strcat(cval,"E");
	  strcat(cval,cexp);
	}
      else
	{
	  if (ise<1)
	    {
	      if (ixv<1)
		{
		  ixv=1;
		  ise=ise-1;
		}
	      sprintf(fmt,"%%%d.%df",3+ixv,ixv);
	      ee=ee*pow(10.0,-ise);
	    }
	  else
	    {
	      if (ixv<0)ixv=0;
	      if (ixe<1)ixe=1;
	      sprintf(fmt,"%%%d.%df",3+ixv+ise,ixe);
	    }
	  sprintf(cval,fmt,rval);
	}
    }

  if (cval[0]==' ') 
    {
      strcpy(temp,cval+1);
      strcpy(cval,temp);  /* For positive numbers */
    }
  ret_lv=strlen(cval)-1; 

  irnd = log10(ee/2.0);
  if (irnd>1) /* Round error */
    {
      irnd=nint_derived(pow(10.0,irnd));
      iee=(int)(ee/irnd+0.999)*irnd;
    }
  else
    iee = ee+0.999;  /* Round error up */

  ee=iee;
  ret_le = log10(ee+0.999)+1.0;
    sprintf(fmt,"%%%dd",ret_le);
    sprintf(cerr,fmt,iee);

  *le = ret_le;
  *lv = ret_lv;


  return 0;
}
int nint_derived(double x){
  int i;
  if(x>0.){
    i=(int)(x+0.5);
  }
  else{
    i=(int)(x-0.5);
  }
  return(i);
}

/* ********************************************************* */
/* Check if the pulsar is within a certain region on the sky */
/* ********************************************************* */
int checkBoundary(char *boundary,pulsar *psr)
{
  double psr_lat,psr_long;
  double centre_lat,centre_long;
  double dlon,dlat,a,c,radius;
  double deg2rad = M_PI/180.0;
  char centre_long_str[MAX_STRLEN],centre_lat_str[MAX_STRLEN];
  int type;
  int ip1,ip2,ip3,nread;

  ip1 = getParam("RAJD",psr); 
  ip2 = getParam("DECJD",psr);

  psr_long = psr->param[ip1].shortVal;
  psr_lat  = psr->param[ip2].shortVal;
  nread = sscanf(boundary,"%d %s %s %lf",&type,centre_long_str,centre_lat_str,&radius);
  if (nread!=4)
    {printf("Problem with determining boundary: e.g. 1 19:33 16:00 2\n"); exit(1);}
  if (type==1) /* Format hh:mm:s.ss dd:mm:s.ss */
    {
      centre_long = turn_deg(hms_turn(centre_long_str));
      centre_lat  = turn_deg(dms_turn(centre_lat_str));
    }
  else if (type==2)  /* Format raj_deg decj_deg */
    {
      sscanf(centre_long_str,"%lf",&centre_long);
      sscanf(centre_lat_str,"%lf",&centre_lat);    
    }
  else if (type==3)  /* Format gl gb */
    {
      sscanf(centre_long_str,"%lf",&centre_long);
      sscanf(centre_lat_str,"%lf",&centre_lat);    
      ip1 = getParam("GL",psr);
      ip2 = getParam("GB",psr);
      psr_long = psr->param[ip1].shortVal;
      psr_lat  = psr->param[ip2].shortVal;
    }
  /* Apply the Haversine formula */
  dlon = (psr_long - centre_long)*deg2rad;
  dlat = (psr_lat  - centre_lat)*deg2rad;
  a = pow(sin(dlat/2.0),2) + cos(centre_lat*deg2rad) * 
    cos(psr_lat*deg2rad)*pow(sin(dlon/2.0),2);
  if (a==1)
    c = M_PI;
  else
    c = 2.0 * atan2(sqrt(a),sqrt(1.0-a))/deg2rad;  
  /* Now place the distance in raddist */
  ip3 = getParam("RADDIST",NULL);
  sprintf(psr->param[ip3].val,"%-8.3g",c);
  if (pcat_maxSize[ip3] < strlen(psr->param[ip3].val)) pcat_maxSize[ip3]=strlen(psr->param[ip3].val);
  psr->param[ip3].shortVal = c;
  psr->param[ip3].set1 = 1;
  if (fabs(c) < radius)
    return 0;
  return 1;
}

void checkCatalogue(FILE *fin,char *fname)
{
  char fline[MAX_STRLEN];
  char current_psr[MAX_STRLEN]="unknown";
  int errornum=1;
  int haveF=0;
  int haveP=0;
  int haveJ2000=0;
  int haveEclip=0;
  int haveDM=0;
  int haveDistDM=0;
  int haveDistDM1=0;
  int haveBIN=0;
  int havePB=0;
  int haveOM=0;
  int haveECC=0;
  int haveEPS1=0;
  int haveEPS2=0;
  int bintype=0;
  int line=0;

  printf("\n\nCHECKING CATALOGUE >%s<\n\n\n",fname);

  while (!feof(fin))
    {
      fgets(fline,MAX_STRLEN,fin);
      line++;
      /* attempt to obtain a label, value, reference and error */
      if (fline[0]=='@') /* Starting a new pulsar */
	{
	  /* check P and F mix */
	  if (haveP==1 && haveF==1)
	    printf("%4d. [ERROR06] Mixture of F's and P's for %s\n",errornum++,current_psr);

	  /* Check mix of J2000 and Ecliptic coordinates */
	  if (haveJ2000 && haveEclip)
	    {
	    printf("%4d. [ERROR20] Mixture of J2000 and Ecliptic coords or PMs for %s\n",errornum++,current_psr);
	    }

	  /* Check existence of DistDM and DistDM1 */
	  if (haveDistDM==0 && haveDM==1)
	    {
	      printf("%4d. [ERROR11] No Dist_dm for %s line: %d\n",errornum++,current_psr,line);
	    }

	  if (haveDistDM1==0 && haveDM==1)
	    {
	      printf("%4d. [ERROR12] No Dist_dm1 for %s line: %d\n",errornum++,current_psr,line);
	    }

	  if (haveBIN==0 && (havePB==1 || haveOM ==1 || haveECC==1))
	      {
	      printf("%4d. [ERROR21] Have PB, OM or ECC but not BINARY for %s\n",errornum++,current_psr);
	      }

	  if (haveOM==1 && haveECC==0)
	      {
		printf("%4d. [ERROR22] Have OM but no ECC for %s\n",errornum++,current_psr);
	      }

	  if (bintype==1 && (haveEPS1==0 || haveEPS2==0))
	      printf("%4d. [ERROR15] ELL1 binary model, but no EPS1 or EPS2 for %s\n",errornum++,current_psr);
	  else if (bintype!=1 && (haveEPS1==1 || haveEPS2==1))
	    printf("%4d. [ERROR17] Not ELL1 binary model, but have EPS1 or EPS2 for %s\n",errornum++,current_psr);
	  if (haveEPS1==1 && haveEPS2==0)
	    printf("%4d. [ERROR18] Have EPS1, but not EPS2 for %s\n",errornum++,current_psr);
	  if (haveEPS1==0 && haveEPS2==1)
	    printf("%4d. [ERROR19] Have EPS2, but not EPS1 for %s\n",errornum++,current_psr);

	  /* Reset flags */
	  haveP=0;
	  haveF=0;
	  haveJ2000=0;
	  haveEclip=0;
	  haveDM=0;
	  haveDistDM=0;
	  haveDistDM1=0;
	  haveBIN=0;
	  havePB=0;
	  haveOM=0;
	  haveECC=0;
	  haveEPS1=0;
	  haveEPS2=0;
	  bintype=0;

	  strcpy(current_psr,"unknown");
	}
      else
	{
	  check_parameter(fline,&haveP,&haveF,&haveJ2000,&haveEclip,&haveDM,&haveDistDM,&haveDistDM1,&haveBIN,&havePB,&haveOM,&haveECC,&haveEPS1,&haveEPS2,current_psr,&bintype,&errornum);
	  /*	  check_reference(fline,&errornum,current_psr);*/
	  check_values(fline,&errornum,current_psr); 
	}
    }
}

/* Check whether the parameter label is correct */

void check_parameter(char* fline,int *haveP,int *haveF,int *haveJ2000,int *haveEclip,int *haveDM,int *haveDistDM,int *haveDistDM1,int *haveBIN, int *havePB, int *haveOM,int *haveECC,int *haveEPS1,int *haveEPS2,char *current_psr,int *bintype,int *errornum)
{
  char label[MAX_STRLEN],name[MAX_STRLEN];
  int ok=0;

  sscanf(fline,"%s",label);
  if (strcmp(label,"PSRJ")==0) 
    {
      ok=1; 
      if (strcmp(current_psr,"unknown")==0) sscanf(fline,"%s %s",label,current_psr);      
      else
	{
	  sscanf(fline,"%s %s",label,name);
	  strcat(current_psr,"/");
	  strcat(current_psr,name);
	}
    }
  else if (strcmp(label,"PSRB")==0) {ok=1; sscanf(fline,"%s %s",label,current_psr);}
  else if (strcmp(label,"P0")==0) {ok=1; check_error(fline,current_psr,errornum); *haveP=1;}
  else if (strcmp(label,"CATALOGUE")==0) {ok=1;}
  else if (strcmp(label,"P1")==0) {ok=1; check_error(fline,current_psr,errornum); *haveP=1;}
  else if (strcmp(label,"F0")==0) {ok=1; check_error(fline,current_psr,errornum); *haveF=1;}
  else if (strcmp(label,"F1")==0) {ok=1; check_error(fline,current_psr,errornum); *haveF=1;}
  else if (strcmp(label,"F2")==0) {ok=1; check_error(fline,current_psr,errornum); *haveF=1;}
  else if (strcmp(label,"DM")==0) {ok=1; check_error(fline,current_psr,errornum); *haveDM=1;}
  else if (strcmp(label,"DM1")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"RM")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"RAJ")==0) {ok=1; check_error(fline,current_psr,errornum); *haveJ2000=1;}
  else if (strcmp(label,"DECJ")==0) {ok=1; check_error(fline,current_psr,errornum); *haveJ2000=1;}
  else if (strcmp(label,"S30")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S40")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S50")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S60")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S80")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S100")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S200")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S300")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S400")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S600")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S700")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S900")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S1400")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S1600")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S3000")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S4000")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S6000")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S8000")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"S10000")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"W50")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"W10")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"PEPOCH")==0) ok=1;
  else if (strcmp(label,"SINI")==0) ok=1;
  else if (strcmp(label,"DM2")==0) ok=1;
  else if (strcmp(label,"POSEPOCH")==0) ok=1;
  else if (strcmp(label,"DMEPOCH")==0) ok=1;
  else if (strcmp(label,"NTOA")==0) ok=1;
  else if (strcmp(label,"CLK")==0) ok=1;
  else if (strcmp(label,"NGLT")==0) ok=1;
  else if (strcmp(label,"TZRMJD")==0) ok=1;
  else if (strcmp(label,"TZRFRQ")==0) ok=1;
  else if (strcmp(label,"TZRSITE")==0) ok=1;
  else if (strcmp(label,"ALIAS")==0) ok=1;
  else if (strcmp(label,"SURVEY")==0) ok=1;
  else if (strcmp(label,"EPHEM")==0) ok=1;
  else if (strcmp(label,"DIST_DM")==0) {ok=1; *haveDistDM=1;}
  else if (strcmp(label,"DIST_DM1")==0) {ok=1; *haveDistDM1=1;}
  else if (strcmp(label,"PMRA")==0) {ok=1; check_error(fline,current_psr,errornum); *haveJ2000=1;}
  else if (strcmp(label,"PMDEC")==0) {ok=1; check_error(fline,current_psr,errornum); *haveJ2000=1;}
  else if (strcmp(label,"TAU_SC")==0) ok=1;
  else if (strcmp(label,"START")==0) ok=1;
  else if (strcmp(label,"FINISH")==0) ok=1;
  else if (strcmp(label,"GLEP")==0) ok=1;
  else if (strcmp(label,"GLF0")==0) ok=1;
  else if (strcmp(label,"GLF1")==0) ok=1;
  else if (strcmp(label,"GLF0D")==0) ok=1;
  else if (strcmp(label,"GLTD")==0) ok=1;
  else if (strcmp(label,"OM")==0) {ok=1; check_error(fline,current_psr,errornum); *haveOM=1;}
  else if (strcmp(label,"OMDOT")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"A1")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"PB")==0) {ok=1; check_error(fline,current_psr,errornum); *havePB=1;}
  else if (strcmp(label,"BINARY")==0) {
    ok=1; 
    *haveBIN=1;
    sscanf(fline,"%s %s",label,name);
    if (strcmp(name,"ELL1")==0)
      *bintype=1;
  }
  else if (strcmp(label,"ECC")==0) {ok=1; check_error(fline,current_psr,errornum); *haveECC=1;}
  else if (strcmp(label,"T0")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"EPS1")==0) {ok=1; check_error(fline,current_psr,errornum); *haveEPS1=1;}
  else if (strcmp(label,"EPS2")==0) {ok=1; check_error(fline,current_psr,errornum); *haveEPS2=1;}
  else if (strcmp(label,"PBDOT")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"TASC")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"A1DOT")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"GAMMA")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"F3")==0) {ok=1; check_error(fline,current_psr,errornum); *haveF=1;}
  else if (strcmp(label,"F4")==0) {ok=1; check_error(fline,current_psr,errornum); *haveF=1;}
  else if (strcmp(label,"F5")==0) {ok=1; check_error(fline,current_psr,errornum); *haveF=1;}
  else if (strcmp(label,"ECCDOT")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"MTOT")==0) ok=1;
  else if (strcmp(label,"XPBDOT")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"M2")==0) ok=1;
  else if (strcmp(label,"NOTE")==0) ok=1;
  else if (strcmp(label,"INTERIM")==0) ok=1;
  else if (strcmp(label,"OM_2")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"OM_3")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"TRES")==0) ok=1;  
  else if (strcmp(label,"NSPAN")==0) ok=1;  
  else if (strcmp(label,"NTCOEF")==0) ok=1;  
  else if (strcmp(label,"NFCOEF")==0) ok=1;  
  else if (strcmp(label,"PREDLEN")==0) ok=1;  
  else if (strcmp(label,"NCOEF")==0) ok=1;  
  else if (strcmp(label,"FBTSUB")==0) ok=1;  
  else if (strcmp(label,"FBNBIN")==0) ok=1;  
  else if (strcmp(label,"FBNCHN")==0) ok=1;  
  else if (strcmp(label,"ECC_2")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"ECC_3")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"A1_2")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"A1_3")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"PB_2")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"PB_3")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"T0_2")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"T0_3")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"ELAT")==0) {ok=1; *haveEclip=1;}
  else if (strcmp(label,"ELONG")==0) {ok=1; *haveEclip=1;}
  else if (strcmp(label,"PMELAT")==0) {ok=1; *haveEclip=1;}
  else if (strcmp(label,"PMELONG")==0) {ok=1; *haveEclip=1;}
  else if (strcmp(label,"PX")==0) {ok=1; check_error(fline,current_psr,errornum);}
  else if (strcmp(label,"SPINDX")==0) ok=1;
  else if (strcmp(label,"TYPE")==0) ok=1;
  else if (strcmp(label,"ASSOC")==0) ok=1;
  else if (strcmp(label,"BINCOMP")==0) ok=1;
  else if (strcmp(label,"DIST_AMN")==0) ok=1;
  else if (strcmp(label,"DIST_A")==0) ok=1;
  else if (strcmp(label,"DIST_AMX")==0) ok=1;
  else if (strcmp(label,"KOM")==0) ok=1;
  else if (strcmp(label,"KIN")==0) ok=1;
  else printf("%4d. [ERROR01] Unknown label: %s for pulsar %s\n",(*errornum)++,label,current_psr);
}

/* Checks the uncertainties */
void check_error(char *fline,char *current_psr,int *errornum)
{
  char label[MAX_STRLEN],error[MAX_STRLEN],val[MAX_STRLEN];
  int nread,i;
  double derror;

  /* See if this line has an error */
  nread = sscanf(fline,"%s %s %s",label,val,error);
  if (nread != 3)
    {
      /*      printf("%4d. [ERROR02] No uncertainty on %s for pulsar %s\n",(*errornum)++,label,current_psr); */
    }
  else
    {
      nread = sscanf(error,"%lf",&derror);
      if (nread==1)
	{
	  for (i=0;i<strlen(error);i++)
	    {
	      if (error[i]=='.')
		printf("%4d. [ERROR04] Non-integral uncertainty on %s for pulsar %s\n",
		       (*errornum)++,label,current_psr);	      
	    }
	  if (derror==0.0)
	    printf("%4d. [ERROR10] Zero uncertainty on %s for pulsar %s\n",
		   (*errornum)++,label,current_psr);
	  /*	  else if (derror>1000.0 && strcmp(label,"T0")!=0 && strcmp(label,"T0_2")!=0 &&
		   strcmp(label,"T0_3")!=0 && strcmp(label,"F2")!=0)
	    printf("%4d. [ERROR16] Huge uncertainty on %s for pulsar %s\n",
	    (*errornum)++,label,current_psr); */
	}
      else
	{
	  if (strcmp(label,"S1400")!=0 && strcmp(label,"S400")!=0 && 
	      strcmp(label,"S600")!=0 && strcmp(label,"S900")!=0 &&
	      strcmp(label,"S700")!=0 && strcmp(label,"S3000")!=0
	      && strcmp(label,"S30")!=0
	      && strcmp(label,"S40")!=0
	      && strcmp(label,"S50")!=0
	      && strcmp(label,"S60")!=0
	      && strcmp(label,"S80")!=0
	      && strcmp(label,"S200")!=0
	      && strcmp(label,"S300")!=0
	      && strcmp(label,"S4000")!=0
	      && strcmp(label,"S6000")!=0
	      && strcmp(label,"S8000")!=0
	      && strcmp(label,"S10000")!=0
	      & strcmp(label,"S100")!=0 && strcmp(label,"S1600")!=0  && strcmp(label,"W10")!=0
	      && strcmp(label,"W50")!=0)
	    printf("%4d. [ERROR03] Something wrong with uncertainty on %s for pulsar %s\n",
		   (*errornum)++,label,current_psr);	  
	}
    }
}

/* Check existence of reference */
void check_reference(char* fline,int *errornum,char *current_psr)
{
  char label[MAX_STRLEN],val[MAX_STRLEN],str1[MAX_STRLEN],str2[MAX_STRLEN];
  char ref[MAX_STRLEN];
  int nread,error,ok;

  ok=0;
  nread = sscanf(fline,"%s %s %s %s",label,val,str1,str2);
  if (nread==3)
    {
      strcpy(ref,str1);
      nread = sscanf(ref,"%d",&error);
      if (nread!=1)
	ok=1;
    }
  else if (nread==4)
    {
      strcpy(ref,str2);
      ok=1;
    }
  if (ok==0)
    {
      if (strcmp(label,"START")!=0 && strcmp(label,"CATALOGUE")!=0 && 
	  strcmp(label,"FINISH")!=0 && strcmp(label,"CLK")!=0
	  && strcmp(label,"EPHEM")!=0 && strcmp(label,"BINARY")!=0 && strcmp(label,"TZRMJD")!=0
	  && strcmp(label,"TZRFRQ")!=0 && strcmp(label,"TZRSITE")!=0 && strcmp(label,"SURVEY")!=0
	  && strcmp(label,"ASSOC")!=0 && strcmp(label,"ALIAS")!=0 && strcmp(label,"PEPOCH")!=0
          && strcmp(label,"BINCOMP")!=0
	  && strcmp(label,"POSEPOCH")!=0 && strcmp(label,"DMEPOCH")!=0 && strcmp(label,"NTOA")!=0 && strcmp(label,"TRES")!=0
	  && strcmp(label,"PSRJ")!=0 && strcmp(label,"TYPE")!=0 && strcmp(label,"DIST_A")!=0
	  && strcmp(label,"NGLT")!=0 && strcmp(label,"INTERIM")!=0  && strcmp(label,"NSPAN")!=0 
	  && strcmp(label,"NCOEF")!=0 && strcmp(label,"NOTE")!=0  && strcmp(label,"NITS")!=0)
	printf("%4d. [ERROR05] No reference on %s for pulsar %s\n",(*errornum)++,label,current_psr);
    }
}

/* Check sensible values */
void check_values(char *fline,int *errornum,char *current_psr)
{
  char label[MAX_STRLEN],valuestr[MAX_STRLEN];
  double value;
  int nread,i;

      nread = sscanf(fline,"%s %lf",label,&value);

  if (strcmp(label,"CLK")!=0 && strcmp(label,"SURVEY")!=0 && strcmp(label,"TYPE")!=0
      && strcmp(label,"PSRJ")!=0 && strcmp(label,"PSRB")!=0 && strcmp(label,"EPHEM")!=0 
      && strcmp(label,"CATALOGUE")!=0 && strcmp(label,"BINARY")!=0 
      && strcmp(label,"ALIAS")!=0 && strcmp(label,"ASSOC")!=0 && strcmp(label,"BINCOMP")!=0 && strcmp(label,"INTERIM")!=0
      && strcmp(label,"NOTE")!=0 && strcmp(label,"TZRSITE")!=0)
    {
      if (nread!=2)
	printf("%4d. [ERROR07] Value for %s for pulsar %s not a number\n",(*errornum)++,
	       label,current_psr);
      else
	{
	  if (strcmp(label,"P1")==0 && (value > 1e-8 || value < 1e-22) && value > 0)
		printf("%4d. [ERROR08] P1 Value for pulsar %s out of range\n",(*errornum)++,current_psr);
	  if (strcmp(label,"P0")==0 && (value < 0 || value > 20))
		printf("%4d. [ERROR09] P0 Value for pulsar %s out of range\n",(*errornum)++,current_psr);
	  if (strcmp(label,"PEPOCH")==0 && (value < 39656 || value > 55197))
	    printf("%4d. [ERROR13] PEPOCH Value is not an MJD for pulsar %s\n",(*errornum)++,current_psr);
	  sscanf(fline,"%s %s",label,valuestr);
	  for (i=0;i<strlen(valuestr);i++)
	    {
	      if (valuestr[i]=='.')
		{
		  if (i==strlen(valuestr-1) || valuestr[i+1]<48 || valuestr[i+1]>57)
		    printf("%4d. [ERROR14] Value for %s for pulsar %s ends with decimal point\n",(*errornum)++,label,current_psr);
		}
	    }
	}
    }
}
