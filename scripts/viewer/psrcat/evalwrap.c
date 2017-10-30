/*
 EVALWRAP.C  Version 1.1

 evaluateExpression: A Simple Expression Evaluator
 Copyright (c) 1996 - 1999 Parsifal Software, All
 Rights Reserved.

 The EVALWRAP.C module provides support functions for the parser function
 evalKernel(), defined by EVALKERN.SYN. It includes definitions of the
 functions called by evalKernel() as well as the definition of the
 evaluateExpression function, implemented as a wrapper function for
 evalKernel().

 This module consists of six parts:
   1. Error diagnostic procedures, including the definition of
      checkZero.
   2. Character stack procedures, including the definition of
      pushChar.
   3. Symbol table procedures, including the definition of
      locateVariable, which provides access to named variables. In this
      implementation, there are no predefined variables. If a variable
      is not found, it is added to the table and initialized to zero.
      The lookup uses a binary search.
   4. Argument stack procedures, including the definition of
      pushArg.
   5. Function call interface which provides access to
      the standard C library math functions.
      The interface consists of
      . a functionTable, each entry of which contains the name of a
        function and a wrapper function which calls the named
        function.
      . an implementation of callFunction which does a binary search of
        functionTable and then calls the appropriate wrapper function.
      Macros are used to simplify generation of the wrapper functions
      and the functionTable entries.
   6. Definition of the evaluateExpression wrapper function.

 For further information about this module, contact
   Parsifal Software
   http://www.parsifalsoft.com
   info@parsifalsoft.com
   1-800-879-2755, 1-508-358-2564
   P.O. Box 219
   Wayland, MA 01778
*/

#include <math.h>
#include <string.h>

#include "evaldefs.h"
#include "evalkern.h"

double exist(double val);
double range(double compare,double val1,double val2);
double ln(double val);
double cosd(double val);
double sind(double val);
double tand(double val);
double sqr(double val);
double sqrt(double val);


/*********************************************************************

 Part 1. Error Diagnostics

*********************************************************************/

ErrorRecord errorRecord;           /* define an error record */

void diagnoseError(char *msg) {
  if (evalKernel_pcb.exit_flag == AG_RUNNING_CODE)   /* parser still running */
    evalKernel_pcb.exit_flag = AG_SEMANTIC_ERROR_CODE;      /* stop parse */
  errorRecord.message = msg;
  errorRecord.line    = evalKernel_pcb.line;
  errorRecord.column  = evalKernel_pcb.column;
}

double checkZero(double value) {
  if (value) return value;
  diagnoseError("Divide by Zero");
  return 1;
}


/*******************************************************************

Part 2. Accumulate variable names and function names

*******************************************************************/

static char  charStack[CHAR_STACK_LENGTH+1];
static char *charStackTop = charStack;

static void resetCharStack(void) {
  charStackTop = charStack;
}

void pushChar(int c) {              /* append char to name string */
  if (charStackTop < charStack+CHAR_STACK_LENGTH) {
    *charStackTop++ = (char) c;
    return;
  }
  /* buffer overflow, kill parse and issue diagnostic */
  diagnoseError("Character Stack Overflow");
}

static char *popString(int nChars) {                /* get string */
  *charStackTop = 0;
  return charStackTop -= nChars;
}


/**********************************************************************

Part 3. Symbol Table

**********************************************************************/

VariableDescriptor variable[N_VARIABLES];    /* Symbol table array */

int nVariables = 0;                       /* no. of entries in table */

/* Callback function to locate named variable */

double *locateVariable(int nameLength) {   /* identify variable name */
  char *name = popString(nameLength);
  int first = 0;
  int last = nVariables - 1;

  while (first <= last) {                           /* binary search */
    int middle = (first+last)/2;
    int flag = strcmp(name,variable[middle].name);
    if (flag == 0) return &variable[middle].value;
    if (flag < 0) last = middle-1;
    else first = middle+1;
  }
  /* name not found, check for room in table */
  if (nVariables >= N_VARIABLES) {
    /* table is full, kill parse and issue diagnostic */
    static double junk = 0;
    diagnoseError("Symbol Table Full");
    return &junk;
  }

  /* insert variable in table in sorted order */
  memmove(&variable[first+1],
          &variable[first],
          (nVariables-first)*sizeof(VariableDescriptor));
  nVariables++;
  variable[first].name = strdup(name);
  variable[first].value = 0;
  return &variable[first].value;
}


/*******************************************************************

Part 4. Accumulate list of function arguments

*******************************************************************/

static double  argStack[ARG_STACK_LENGTH];      /* argument buffer */
static double *argStackTop = argStack;

static void resetArgStack(void) {
  argStackTop = argStack;
}

void pushArg(double x) {                     /* store arg in list */
  if (argStackTop < argStack + ARG_STACK_LENGTH) {
    *argStackTop++ = x;
    return;
  }
  /* too many args, kill parse and issue diagnostic */
  diagnoseError("Argument Stack Full");
}

static double *popArgs(int nArgs) {                 /* fetch args */
  return argStackTop -= nArgs;
}


/**********************************************************************

 Part 5. Function Call Interface

 Define functionTable, each entry of which contains the ascii name of
 a function and a pointer to a wrapper function. The wrapper function
 checks the argument count and calls the real function.

 Then, define callFunction. Given the ascii name of a function,
 callFunction does a binary search of functionTable and on a successful
 search calls the corresponding wrapper function.

**********************************************************************/

/* define some macros to build the wrapper functions */

/*
 First, a macro to make a wrapper function for a function with one
 argument.
*/

#define WRAPPER_FUNCTION_1_ARG(FUN) \
double FUN##Wrapper(int argc, double *argv) {\
  if (argc == 1) return FUN(argv[0]);\
  diagnoseError("Wrong Number of Arguments");\
  return 0;\
}

/*
 Now, a macro to make a wrapper function for a function with two
 arguments.
*/

#define WRAPPER_FUNCTION_2_ARGS(FUN) \
double FUN##Wrapper(int argc, double *argv) {\
  if (argc==2) return FUN(argv[0], argv[1]);\
  diagnoseError("Wrong Number of Arguments");\
  return 0;\
}

/*
 Now, a macro to make a wrapper function for a function with two
 arguments.
*/

#define WRAPPER_FUNCTION_3_ARGS(FUN) \
double FUN##Wrapper(int argc, double *argv) {\
  if (argc==3) return FUN(argv[0], argv[1],argv[2]);\
  diagnoseError("Wrong Number of Arguments");\
  return 0;\
}


/*
 Now define wrapper functions for the standard C library
 math functions.
*/

WRAPPER_FUNCTION_1_ARG(acos)
WRAPPER_FUNCTION_1_ARG(asin)
WRAPPER_FUNCTION_1_ARG(atan)
WRAPPER_FUNCTION_2_ARGS(atan2)
WRAPPER_FUNCTION_1_ARG(cos)
WRAPPER_FUNCTION_1_ARG(cosd)
WRAPPER_FUNCTION_1_ARG(cosh)
WRAPPER_FUNCTION_1_ARG(exist) 
WRAPPER_FUNCTION_1_ARG(exp)
WRAPPER_FUNCTION_1_ARG(fabs)
WRAPPER_FUNCTION_2_ARGS(fmod)
WRAPPER_FUNCTION_1_ARG(ln)
WRAPPER_FUNCTION_1_ARG(log10)
WRAPPER_FUNCTION_3_ARGS(range)
WRAPPER_FUNCTION_1_ARG(sin)
WRAPPER_FUNCTION_1_ARG(sind)
WRAPPER_FUNCTION_1_ARG(sinh)
WRAPPER_FUNCTION_1_ARG(sqr)
WRAPPER_FUNCTION_1_ARG(sqrt)
WRAPPER_FUNCTION_1_ARG(tan)
WRAPPER_FUNCTION_1_ARG(tand)
WRAPPER_FUNCTION_1_ARG(tanh)




/* A macro to make correct functionTable entries */
#define TABLE_ENTRY(FUN) {#FUN, FUN##Wrapper}

/* remember to fix this when you add more functions to the table */
#define N_FUNCTIONS 22

/* define the function table -- must be in sorted order! */
struct {
  char *name;
  double (*function)(int, double[]);
} functionTable[N_FUNCTIONS] = {
  TABLE_ENTRY(acos),
  TABLE_ENTRY(asin),
  TABLE_ENTRY(atan),
  TABLE_ENTRY(atan2),
  TABLE_ENTRY(cos),
  TABLE_ENTRY(cosd),
  TABLE_ENTRY(cosh),
  TABLE_ENTRY(exist), 
  TABLE_ENTRY(exp),
  TABLE_ENTRY(fabs),
  TABLE_ENTRY(fmod),
  TABLE_ENTRY(ln),
  TABLE_ENTRY(log10),
  TABLE_ENTRY(range),
  TABLE_ENTRY(sin),
  TABLE_ENTRY(sind),
  TABLE_ENTRY(sinh),
  TABLE_ENTRY(sqr),
  TABLE_ENTRY(sqrt),
  TABLE_ENTRY(tan),
  TABLE_ENTRY(tand),
  TABLE_ENTRY(tanh),
};

double range(double compare,double val1,double val2)
{
  if (val2 < val1)
    {
      if (val1 < compare)
	return 1;
      if (val2 > compare)
	return 1;    
    }
  else
    {
      if (val1 < compare && val2 > compare)
	return 1;
    }
  return 0;
}

double exist(double val)
{
  return 1.0;
}

double ln(double val)
{
  return log(val);
}

double cosd(double val)
{
  return cos(val*M_PI/180.0);
}

double sind(double val)
{
  return sin(val*M_PI/180.0);
}

double sqr(double val)
{
  return val*val;
}


double tand(double val)
{
  return tan(val*M_PI/180.0);
}

/* Finally, define the callback function to perform a function call */

double callFunction(int nameLength, int argCount) {
  char *name = popString(nameLength);
  double *argValues = popArgs(argCount);
  int first = 0;
  int last = N_FUNCTIONS-1;
  while (first <= last) {                     /* binary search */
    int middle = (first+last)/2;
    int flag = strcmp(name,functionTable[middle].name);
    if (flag == 0) return functionTable[middle].function(argCount, argValues);
    if (flag < 0) last = middle-1;
    else first = middle+1;
  }
  diagnoseError("Unknown Function");
  return 0;
}

/*******************************************************************

Part 6. Wrapper function definition

*******************************************************************/

int evaluateExpression(char *expressionString) {
  resetCharStack();
  resetArgStack();
  evalKernel_pcb.pointer = (unsigned char *) expressionString;
  evalKernel();
  return evalKernel_pcb.exit_flag != AG_SUCCESS_CODE;
}

/* End of evalwrap.c */
