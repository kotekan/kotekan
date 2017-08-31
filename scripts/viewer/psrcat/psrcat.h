/* The following should be the CVS version number of psrcat.c */
#define PSRCAT_SOFTWARE_VERSION 1.49
 
#define MAX_REFLEN 20      /* Maximum length for references */
#define MAX_PARAM  250     /* Maximum number of parameters   */
#define MAX_PSR    4096    /* Maximum number of pulsars (for optimal speed keep to a power of 2) */
#define MAX_STRLEN 100     /* Maximum length for general strings */
#define MAX_PLIST  80      /* Maximum number of parameters that can be printed */
#define MAX_NAMELEN 20     /* Maximum length of a pulsar name */
#define MAX_FILTERSTR 500  /* Maximum string length for the filter */
#define MAX_HEADERLEN 2000 /* Maximum length for the header string */
#define MAX_CATLEN 100     /* Maximum length of catalogue path and name */
#define MAX_MERGELEN 500   /* Maximum length of merge path and name */
#define MAX_SURVEYS 50     /* Maximum number of different surveys */
#define MAX_HELPLEN 200    /* Maximum length for help strings */
#define MAX_ALIASES 200    /* Maximum number of aliases */

#define NO_ERR "0"             /* NULL string for no error */
#define NO_REF "*"              /* NULL string for no reference */

#define CUSTOM_FORMAT "%-16.4e" /* Formatting string for custom values */

#define MASS_PSR 1.35      /* Typical mass of pulsar in solar masses */
#define PSR_INERTIA 1.0e45 /* Typical moment of inertia for a pulsar */
#define oneAU 149597870    /* 1 AU in km */
#define onePC 30.857e12    /* 1 pc in km */

/* HTML page if click on pulsar name -- only required for WEB */
#define COUNTER_FILE "/nfs/wwwatdocs/people/George.Hobbs/catalogue/counter.dat"
#define PSRNAME_HTML "http://www.atnf.csiro.au/people/ghobbs/link_name2.html"

typedef struct parameter
{
  char   val[MAX_STRLEN];
  double shortVal;
  char   err[MAX_REFLEN];
  double error_expand;
  char   ref[MAX_REFLEN];
  int    set1;             /* = 1 if value has been set */
  int    set2;             /* = 1 if error has been set */  
  int    set3;             /* = 1 if reference has been set */
  int    derived;          /* = 1 if parameter has been derived */
} parameter;

typedef struct paramtype
{
  char header1[MAX_STRLEN]; /* String for top line of header */
  char header2[MAX_STRLEN]; /* String for bottom line of header */
  char shortStr[MAX_STRLEN];/* String for format statement for SHORT output */
  char help[MAX_HELPLEN];   /* String for help output */
  int  numeric;             /* = 0 if alphabetic, 1 = numeric */
  int  helpType;            /* Number stating which group this parameter belongs to for printing */
  int  ephem;               /* = 1 if used in TEMPO ephemeris */
} paramtype;

typedef struct pulsar
{
  parameter param[MAX_PARAM]; 
  int merged;
} pulsar;

typedef struct linkedList
{
  pulsar *current;
  struct linkedList *post;
}linkedList;

/* Function definitions */
int readCatalogue(pulsar *psr,linkedList *list,int *npsr,char name[MAX_PSR][MAX_NAMELEN],int iname,
		   int *haveName,char *filterStr,int *ptr,char sptr[MAX_PLIST][MAX_STRLEN],int np,char *dbFile,
		   int merge,int willmerge,char *boundary,paramtype *defParam,int checkCat,
		  char alias1[MAX_ALIASES][MAX_STRLEN],char alias2[MAX_ALIASES][MAX_STRLEN],int aliasCount,int exactMatch);
void displayOutput(pulsar *psr,linkedList *list,int npsr,int *ptr,int np,paramtype *defParam,int nohead,
		   int tbForm, int noNumber,int webflag);
int  getParam(char *str,pulsar *psr);
void parseParameters(int argc,char *argv[],int *ptr,char sptr[MAX_PLIST][MAX_STRLEN],
		     int *np,char name[MAX_PSR][MAX_NAMELEN],int *iname,
		     int *haveName,char *filterStr,char *customStr,int *nohead,int *tbForm,int *noNumber,
		     char *dbFile,char *mergeCat,paramtype *defParam,int *ephem,char *boundary,
		     int *allmerge,int *webflag,char *plotx,char *ploty,int *plotx_log,
		     int *ploty_log,int *bib_info,int *listref,int *listallRef,
		     int *listparticularRef,char *listIndRef,int *checkCat,int *fsize,
		     int *nocand,int *nointerim,int *exactMatch);
void deriveParameter(char *str,pulsar *psr,int ip);
int rnd8(double rval,double rerr,int ifac,char *cval,int *lv,char *cerr,int *le,char *msg);
void plotParams(char *plotx,char *ploty,int plotx_log,int ploty_log,pulsar *psr,
		linkedList *list,int npsr);
double formExpression(char *filterStr,pulsar *psr,int *badval,int exactMatch);
double hms_turn(char *line);
double dms_turn(char *line);
double turn_deg(double turn);
int turn_dms(double turn, char *dms);
int turn_hms(double turn, char *hms);
double errscale(char * valstr, char * errstr);
void upperCase(char *str);
void defineParameters(paramtype *defParam);
int find_galcoord_equatorial(double ra2000,double dec2000,double pmra,double pmraerr,double pmdec,double pmdecerr, double dist,double *mul,double *mulerr,double *mub,double *muberr);
int find_galcoord_ecliptic(double el,double eb,double pmel,double pmelerr,double pmeb,double pmeberr, double dist,double *mul,double *mulerr,double *mub,double *muberr);



int  pcat_expert;
char pcat_parameters[MAX_PARAM][20];
int  pcat_maxSize[MAX_PARAM];
int  pcat_refZero[MAX_PARAM];
int  pcat_errZero[MAX_PARAM];
int  pcat_sort;
int  pcat_descend;
char pcat_custom1[MAX_FILTERSTR];
char pcat_custom2[MAX_FILTERSTR];
char pcat_custom3[MAX_FILTERSTR];
char pcat_custom4[MAX_FILTERSTR];
int  verbose;
char NO_VAL[MAX_STRLEN];  /* NULL string for no value */


