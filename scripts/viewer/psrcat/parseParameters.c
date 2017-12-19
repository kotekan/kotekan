#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "psrcat.h"


void help();
void version(char *dbFile);
void display_parameters(char *checkval,paramtype *defParam);
void parseParameters(int argc,char *argv[],int *ptr,char sptr[MAX_PLIST][MAX_STRLEN],
		     int *np,char name[MAX_PSR][MAX_NAMELEN],int *iname,
		     int *haveName,char *filterStr,char *customStr,int *nohead,int *tbForm,int *noNumber,
		     char *dbFile,char *mergeCat,paramtype *defParam,int *ephem,char *boundary,
		     int *allmerge,int *webflag,char *plotx,char *ploty,int *plotx_log,int *ploty_log,
		     int *bib_info,int *listRef,int *listallRef,int *listparticularRef,char *listIndRef,
		     int *checkCat,int *fsize,int *nocand,int *nointerim,int *exactMatch)
{
  register unsigned int i,j;
  char *strptr;
  char bound1[1000];
  char bound2[1000];
  pcat_expert = 0;
  strcpy(bound1,""); strcpy(bound2,"");
  *iname = 0; /* Number of pulsar names given on the command line */
  /* Convert to upper case */
  for (i=1;i<argc;i++)
    {
      if (strcasecmp(argv[i-1],"-db_file")!=0  && strcasecmp(argv[i-1],"-psr")!=0 && strcasecmp(argv[i-1],"-merge")!=0 &&
	  strcasecmp(argv[i-1],"-group")!=0 && strcasecmp(argv[i-1],"-web")!=0) /* Don't put filenames in upper case */
	upperCase(argv[i]);
    }
  for (i=1;i<argc;i++)
    {
      if (strcmp(argv[i],"-S")==0)      /* Define sort parameter */
	pcat_sort=getParam(argv[++i],NULL);
      else if (strcmp(argv[i],"-H")==0)  /* Help */
	help();
      else if (strcmp(argv[i],"-V")==0)  /* Version of catalogue */
	version(dbFile);
      else if (strcmp(argv[i],"-LISTREF")==0)    /* List reference */
	*listRef=1;
      else if (strcmp(argv[i],"-NOCAND")==0) /* No candidates */
	*nocand=1;
      else if (strcmp(argv[i],"-NOINTERIM")==0) /* No interim */
	*nointerim=1;
      else if (strcmp(argv[i],"-BBL")==0)    /* List reference */
	*listRef=2;
      else if (strcmp(argv[i],"-ALLBIBS")==0)    /* List reference */
	*listRef=3;
      else if (strcmp(argv[i],"-ALLREF")==0)     /* List all references */
	*listallRef=1;
      else if (strcmp(argv[i],"-EXACTMATCH")==0)     /* Exact matching required */
	*exactMatch=1;
      else if (strcmp(argv[i],"-BIB")==0)        /* Provide bibliographic reference */
	{
	  *listparticularRef=1;
	  strcpy(listIndRef,argv[++i]);
	}
      else if (strcmp(argv[i],"-BOUNDARY")==0)   /* Boundary string */
	strcpy(boundary,argv[++i]);
      else if (strcmp(argv[i],"-BOUND_COORD")==0) /* Old boundary method */
	strcpy(bound1,argv[++i]);
      else if (strcmp(argv[i],"-BOUND_RADIUS")==0) /* Old boundary method */
	strcpy(bound2,argv[++i]);
      else if (strcmp(argv[i],"-CHECK")==0) /* Check catalogue */
	*checkCat=1;
      else if (strcmp(argv[i],"-ALL")==0)   /* Merge with PSRCAT_RUNDIR environment variable */
	*allmerge=1;
      else if (strcmp(argv[i],"-E")==0)   /* TEMPO ephemeris */
	*ephem = 1;
      else if (strcmp(argv[i],"-E2")==0)  /* Full ephemeris */
	*ephem = 2;
      else if (strcmp(argv[i],"-E3")==0)  /* Select ephemeris */
	*ephem = 3;
      else if (strcmp(argv[i],"-E4")==0)  /* Select ephemeris */
	*ephem = 4;
      else if (strcmp(argv[i],"-FSIZE")==0)  /* Select ephemeris */
	sscanf(argv[++i],"%d",fsize);
      else if (strcmp(argv[i],"-EXPERT")==0)  /* expert mode */
	pcat_expert = 1;
      else if (strcmp(argv[i],"-DESCEND")==0)  /* Sort in descending order */      
	pcat_descend=1;	
      else if (strcmp(argv[i],"-DB_FILE")==0)  /* Catalogue file */
	strcpy(dbFile,argv[++i]);
      else if (strcmp(argv[i],"-PLOTX")==0)    /* expression for plotting on x-axis */
	{strcpy(plotx,argv[++i]);*nohead=1;}
      else if (strcmp(argv[i],"-PLOTY")==0)    /* ploty */
	{strcpy(ploty,argv[++i]);*nohead=1;}
      else if (strcmp(argv[i],"-LOGX")==0)     /* Take log10 of x-axis */
	*plotx_log = 1;
      else if (strcmp(argv[i],"-LOGY")==0)     /* Take log10 of y-axis */
	*ploty_log = 1;
      else if (strcmp(argv[i],"-MERGE")==0)  /* Merge catalogues */
	strcat(mergeCat,argv[++i]);
      else if (strcmp(argv[i],"-BIBLIOGRAPHY")==0)
	*bib_info=1;
      else if (strcmp(argv[i],"-NONUMBER")==0)  /* No numbering */
	*noNumber=1;
      else if (strcmp(argv[i],"-O")==0)  /* Tabular format */
	{
	  i++;
	  if (strcmp(argv[i],"LONG")==0)           *tbForm=0;
	  if (strcmp(argv[i],"PUBLISH")==0)        *tbForm=1;
	  if (strcmp(argv[i],"SHORT")==0)          *tbForm=2;
	  if (strcmp(argv[i],"SHORT_NO_ERROR")==0) *tbForm=2;
	  if (strcmp(argv[i],"SHORT_ERROR")==0)    *tbForm=3;
	  if (strcmp(argv[i],"LONG_ERROR")==0)     *tbForm=4;
	  if (strcmp(argv[i],"NONE")==0)           *tbForm=5;
	  if (strcmp(argv[i],"SHORT_CSV")==0)      *tbForm=6;
	  if (strcmp(argv[i],"LONG_ERROR_CSV")==0) *tbForm=7;
	  
	}
      else if (strcmp(argv[i],"-C1")==0) /* Custom parameter 1 */	
	  strcat(pcat_custom1,argv[++i]);
      else if (strcmp(argv[i],"-C2")==0) /* Custom parameter 2 */	
	  strcat(pcat_custom2,argv[++i]);
      else if (strcmp(argv[i],"-C3")==0) /* Custom parameter 3 */	
	  strcat(pcat_custom3,argv[++i]);
      else if (strcmp(argv[i],"-C4")==0) /* Custom parameter 4 */	
	  strcat(pcat_custom4,argv[++i]);
      else if (strcmp(argv[i],"-NULL")==0) /* Null value string */
	  strcpy(NO_VAL,argv[++i]);
      else if (strcmp(argv[i],"-WEB")==0) /* Web flag */
	{
	  char web_address[MAX_STRLEN*3];
	  *webflag=1;
	  /* Add to counter */
	  strcpy(web_address,"echo \"");
	  strcat(web_address,argv[i+1]); i++;
	  strcat(web_address,"\" >>");
	  strcat(web_address,COUNTER_FILE);
	  system(web_address);	  
	}
      else if (strcmp(argv[i],"-P")==0) /* Describe parameters */
	{
	  if (i+1==argc)
	    display_parameters("*",defParam);                 /* List all parameters */
	  else
	    display_parameters(argv[i+1],defParam);           /* Get selected parameter list */
	}
      else if (strcmp(argv[i],"-C")==0) /* Define parameters */
	{
	  customStr = argv[i];
	  *np = 0;

	  for (j=0;j<MAX_PLIST;j++)
	    { 	      
	      if (j==0)
		strptr = strtok(argv[i+1]," \t\n");
	      else
		strptr = strtok(NULL," \t\n");		

	      if (strptr==NULL)
		break;
	      if (strcmp(strptr,"JNAME")==0)
		{strcpy(sptr[*np],"PSRJ"); strcpy(strptr,"PSRJ");}
	      else if (strcmp(strptr,"BNAME")==0)
		{strcpy(sptr[*np],"PSRB"); strcpy(strptr,"PSRB");}
	      else
		strcpy(sptr[*np],strptr);
	      ptr[(*np)++] = getParam(strptr,NULL);
	      if (ptr[(*np)-1]==-1)
		{
		  printf("Parameter >%s< not known.  Use psrcat -p to get a list of parameters\n",strptr);
		  exit(1);
		}
	    }
	  i++;
	}
      else if (strcmp(argv[i],"-L")==0) /* Logical expression for filtering pulsars */
	  strcat(filterStr,argv[++i]);
      else if (strcmp(argv[i],"-NOHEAD")==0)
	*nohead=1;
      else if (strcmp(argv[i],"-VERBOSE")==0)
	verbose=1;
      else if (strcmp(argv[i],"-X")==0) /* Simplify some settings */
	{
	  *nohead=1;
	  *noNumber=1;
	  *tbForm=4;
	}
      else if (strcmp(argv[i],"-PSR")==0) /* Read pulsar names from a file */
	{
	  FILE *fin;
	  char temp[1000];
	  i++;
	  if (!(fin = fopen(argv[i],"r")))
	    {
	      printf("Unable to open pulsar file %s\n",argv[i]);
	      exit(1);
	    }
	  while (!feof(fin))
	    {
	      if (fscanf(fin,"%s",temp)==1) {
		haveName[*iname]=0; 
		if (temp[0] == '0' || temp[0]=='1' || temp[0] =='2')
		  sprintf(name[(*iname)++],"?%s",temp);
		else
		  sprintf(name[(*iname)++],"%s",temp);
	      }
	    }
	  fclose(fin);
	}
      else if (argv[i][0] == '0' || argv[i][0] == '1' || argv[i][0] == '2') /* Pulsar name */
	{haveName[*iname] = 0; sprintf(name[(*iname)++],"?%s",argv[i]);}
      else if (argv[i][0] == 'B' || argv[i][0] == 'J') /* Pulsar name */
	{haveName[*iname] = 0; sprintf(name[(*iname)++],"%s",argv[i]);}
      else
	{
	  printf("Unknown parameter >%s<.  Use -h to get list of parameters.\n",argv[i]);
	  exit(1);
	}
      
    }
  if (strlen(bound1)>0 && strlen(bound2)>0)
    {
      strcpy(boundary,bound1);
      strcat(boundary," ");
      strcat(boundary,bound2);
    }

}
void upperCase(char *str)
{
  register unsigned int i;

  for (i=strlen(str);i--;)
    str[i] = toupper(str[i]);
}


void help()
{
  printf("-all            Merge with $PSRCAT_RUNDIR/obs*.db\n");
  printf("-allref         Lists the entire bibliographic reference file\n");
  printf("-bib            Provide bibliographic reference\n");
  printf("-bibliography   Lists references within the catalogue\n");
  printf("-c              List of parameters to obtain from the database\n");
  printf("-check          Checks the catalogue for incorrect entries\n");
  printf("-db_file        Database path and name\n");
  printf("-descend        Sort in descending order\n");
  printf("-e              Produce short ephemeris files suitable for TEMPO\n");
  printf("-e2             Produce long ephemeris files\n");
  printf("-e3             Produce ephemeris file with selected parameters\n");
  printf("-exactMatch     Require exact match for string comparisons\n");
  printf("-h              This help information\n"); 
  printf("-l              Logical string for filtering unwanted pulsars\n");
  printf("                e.g. p0 > 5\n");
  printf("-listref        Provide bibliographic references at end of table\n");
  printf("-merge          Merges multiple catalogue files.  For example\n");
  printf("                -merge \"mycat*.db\" will merge the public catalogue\n");
  printf("                with all the catalogues with filenames mycat*.db in the\n");
  printf("                local directory.\n");
  printf("-nohead         Do not print a header at the top of the table\n");
  printf("-nonumber       Do not print line numbers in output\n");
  printf("-nocand         Do not include pulsar candidates\n");
  printf("-nointerim      Do not include interim solutions\n");
  printf("-null           Null string for displaying for unknown parameter\n");
  printf("-o              Output style: short\n");
  printf("                              short_error\n");
  printf("                              long [default]\n");
  printf("                              long_error\n");
  printf("                              publish\n");
  printf("                              short_csv\n");
  printf("                              long_error_csv\n");
  printf("-p <match>      Display help on all available parameters. If 'match' is present then\n"); 
  printf("                only lines containing the string 'match' are displayed\n");
  printf("-psr filename   Reads pulsar names from file 'filename'\n");
  printf("-s              Parameter for sorting\n");
  printf("-v              Display version number of the catalogue\n");
  printf("-verbose        Prints warning messages - use to check entries in catalogue\n");
  printf("-x              Quick method for -nohead -nonumber -o long_error\n");
  exit(1);
}

/* List all the available parameters */
void display_parameters(char *checkval,paramtype *defParam)
{
  int i,j,count,num;
  char upper1[MAX_STRLEN];
  char upper2[MAX_STRLEN];
  if (strcmp(checkval,"*")==0)
    {
      count=1;
      for (num=1;num<13;num++)
	{
	  if (num==1)
	    printf("\nNames and astrometric parameters:\n\n");
	  else if (num==2)
	    printf("\nRotational, dispersion measure and pulse shape parameters:\n\n");
	  else if (num==3)
	    printf("\nBinary parameters:\n\n");
	  else if (num==4)
	    printf("\nDistance parameters:\n\n");
	  else if (num==5)
	    printf("\nSurvey and pulsar type parameters:\n\n");
	  else if (num==6)
	    printf("\nDerived parameters:\n\n");
	  else if (num==7)
	    printf("\nTiming solution parameters:\n\n");
	  else if (num==8)
	    printf("\nWide-band correlator parameters:\n\n");
	  else if (num==9)
	    printf("\nFilter-bank parameters:\n\n");
	  else if (num==10)
	    printf("\nCPSR parameters:\n\n");
	  else if (num==11)
	    printf("\nCustom parameters:\n\n");
	  else if (num==12)
	    printf("\nGlitch parameters:\n\n");
	  for (i=0;i<MAX_PARAM;i++)
	    {
	      if (strlen(pcat_parameters[i])>0 && defParam[i].helpType == num)
		{
		  printf("%5d. %s\t\t%s\n",count++,pcat_parameters[i],defParam[i].help);	  
		}
	    }
	}
    }
  else
    {
      count=1;
      for (i=0;i<MAX_PARAM;i++)
	{
	  strcpy(upper1,pcat_parameters[i]); 
	  strcpy(upper2,defParam[i].help); 
	  upperCase(upper1);
	  upperCase(upper2);

	  if (strstr(upper1,checkval)!=NULL || strstr(upper2,checkval)!=NULL)
	    printf("%5d. %s\t\t%s\n",count++,pcat_parameters[i],defParam[i].help);
	}
    }
  exit(1);
}


/* Open catalogue and display version number */
void version(char *dbFile)
{
  char *path;
  char str[1000];
  FILE *fin;
  char* found;


  printf("Software version: %.2f\n",PSRCAT_SOFTWARE_VERSION);

  if (strlen(dbFile)==0)
    {
      path = getenv("PSRCAT_FILE");

      if (path == NULL) /* Use the default database */
	strcpy(dbFile,"/pulsar/psr/runtime/psrcat/psrcat.db");
      else
	strcpy(dbFile,path);
    }
  if (!(fin = fopen(dbFile,"r")))
    {
      printf("PSRCAT: Unable to open catalogue file: %s\n",dbFile);
      exit(1);
    }
  while (!feof(fin))
    {
      fscanf(fin,"%s",str);
      found = strstr(str, "CATALOGUE");
      
      if (found != 0)
	{
	  fscanf(fin,"%s",str);
	  printf("Catalogue version number = %s\n",str);
	  exit(1);
	}
    }
  printf("No catalogue version number available\n");
  exit(1);
}

