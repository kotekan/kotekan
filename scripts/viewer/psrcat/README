This package contains the source code of "psrcat" and the public
catalogue (psrcat.db).  This README file describes how to compile
PSRCAT on Solaris, Linux and Mac OS machines and also explains how the
catalogue can be accessed externally from a separate program.

To unpack the distribution

> gunzip psrcat_pkg.tar.gz
> tar -xvf psrcat_pkg.tar

This will make a directory called psrcat_tar containing the source files
and the public catalogue.  

To produce the binary (checked on linux with gcc v2.95.4, solaris and Mac OS X 10.3):

> cd psrcat_tar
> source makeit

This will produce a binary file "psrcat" which can be run as follows:

> psrcat -db_file psrcat.db

For help type:

> psrcat -h

Examples:

1. List the names, periods and dispersion measures for all pulsars with periods > 3

     > psrcat -db_file psrcat.db -c "name p0 dm" -l "p0 > 3"

2. Define a custom variable (sqrt(p0*p1)) and sort on this parameter in ascending order

    > psrcat -db_file "psrcat.db" -c "name p0 c1" -c1 "sqrt(p0*p1)" -s "c1"

3. Print period and period derivative in the short format

    > psrcat -db_file "psrcat.db" -c "p0 p1" -o short

Note:

It is possible to define the environment variable $PSRCAT_FILE to point to the
catalogue instead of using -db_file.

-------------------------------------------------------------------------------
Calling the catalogue from a C program

It is possible to obtain parameters from the catalogue using the function
callPsrcat_val or callPsrcat_string. 

int callPsrcat_val(char *cat,char *psr,char *label,double *val,double *err,char *ref)
int callPsrcat_string(char *cat,char *psr,char *label,char *str)

These functions return 1 if the catalogue cannot be opened
	               2 if the pulsar is not known
		       3 if the parameter is not known
  		       0 Otherwise

Both functions have as input:

cat = name of catalogue ("public" defaults to the public catalogue set by the
      environment variable PSRCAT_FILE)
psr = name of pulsar
label = name of parameter

and callPsrcat_val returns:

val = value of parameter
err = uncertainty on the parameter
ref = reference given to the parameter

and callPsrcat_string returns:

str = string representing the parameter.

Example:

#include <stdio.h>

int main()
{
 int ret;
 double val,err;
 char ref[100],str[100];

 ret = callPsrcat_val("public","b1933+16","p0",&val,&err,ref);
 printf("B1933+16 has a period of %f+/-%f\n",val,err);
 ret = callPsrcat_val("public","b0329+54","dm",&val,&err,ref);
 printf("and B0329+54 has a dispersion measure of %f+/-%f\n",val,err);
 ret = callPsrcat_string("public","b0329+54","survey",&str);
 printf("and B0329+54 has been seen in the following surveys: %s\n",str);

 return 0;
}

This program should be compiled with:

On Solaris:
cc -o myoutput externalCall.c myProgram.c  -L./ -lpsrcat -lm -lsunmath -lF77 -lM77

and on Linux:
gcc -o myoutput externalCall.c myProgram.c  -L./ -lpsrcat -lm -lg2c

(where myoutput is the output binary and myProgram.c is your program.)

Notes:

1. The catalogue will be opened and read during the first function call.  It will
   then be held in memory until a new catalogue is required.



