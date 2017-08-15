#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "psrcat.h"
#include <string.h>

void displayOutput(pulsar *psr,linkedList *list,int npsr,int *ptr,int np,paramtype *defParam,
		   int nohead,int tbForm,int noNumber,int webflag)
{
  parameter dispParam;
  register unsigned int i,j,ip,k;
  char dispStr[MAX_STRLEN];
  char headerStr[MAX_HEADERLEN]="";
  char headerStr2[MAX_HEADERLEN]="";
  char temp[MAX_STRLEN],temp2[MAX_STRLEN];
  pulsar *psrptr;
  int plusSign;
  char *strptr;
  register unsigned int space;
  int ival,ierr;

  psrptr = list[0].current;
  /* Display header if required */
  if (nohead==0)
    {
      if (noNumber==0)
	{
	  if (tbForm!=5)
	    {
	      if (tbForm==6 || tbForm==7)
		{
		  strcat(headerStr,"#;");
		  strcat(headerStr2," ;");
		}
	      else
		{
		  strcat(headerStr,"#     ");  /* Counter */
		  strcat(headerStr2,"      "); /* Counter */
		}
	    }
	}
      for (j=0;j<np;j++)
	{
	  /* Room for value */
	  ip = ptr[j];
	  //	  printf("Size = %d %d %d\n",pcat_maxSize[ip],pcat_errZero[ip],pcat_refZero[ip]);
	  if (tbForm!=2 && tbForm!=3 && tbForm!=6 && tbForm!=7)
	    {
	      strcat(headerStr,defParam[ip].header1);
	      strcat(headerStr2,defParam[ip].header2);
	      if (tbForm==4 && pcat_errZero[ip]!=0)
		{
		  strcat(headerStr,"         ");
		  strcat(headerStr2,"         ");
		}

	    }

	  if (tbForm==0 || tbForm==4)      /* long        */
	    {
	      for (i=strlen(defParam[ip].header1);i<=pcat_maxSize[ip];i++) strcat(headerStr," "); 
	    }
	  else if (tbForm==1) /* publication */
	    for (i=strlen(defParam[ip].header1);i<=pcat_maxSize[ip]+4;i++) strcat(headerStr," "); 
	  else if (tbForm==2 || tbForm==3 || tbForm==6 || tbForm==7) /* short       */
	    {
	      strcpy(temp,defParam[ptr[j]].shortStr);
	      strptr = strtok(temp,"%.-sgef");
	      sscanf(strptr,"%d",&ival);
	      if (strstr(defParam[ptr[j]].shortStr,"-")!=NULL)
		{
		  strcat(headerStr,defParam[ip].header1);
		  strcat(headerStr2,defParam[ip].header2);
		  if (tbForm==3) ival--;
		  if (tbForm!=6 && tbForm!=7)
		    {
		      for (i=strlen(defParam[ip].header1);i<ival+2;i++) strcat(headerStr," ");	      
		      for (i=strlen(defParam[ip].header2);i<ival+2;i++) strcat(headerStr2," ");	      
		    }
		  else
		    {
		      strcat(headerStr,";");
		      strcat(headerStr2,";");
		    }
		}
	      else
		{
		  if (tbForm==6 || tbForm==7)
		    {
		      strcat(headerStr,defParam[ip].header1);
		      strcat(headerStr2,defParam[ip].header2);
		      strcat(headerStr,";");
		      strcat(headerStr2,";");
		    }
		  else
		    {
		      for (i=strlen(defParam[ip].header1);i<ival;i++) strcat(headerStr," ");	      
		      for (i=strlen(defParam[ip].header2);i<ival;i++) strcat(headerStr2," ");	      
		      strcat(headerStr,defParam[ip].header1);
		      strcat(headerStr2,defParam[ip].header2);
		      strcat(headerStr,"  ");
		      strcat(headerStr2,"  ");
		    }
		}
	    }
	
	  if (tbForm==0 || tbForm==4)
	    for (i=strlen(defParam[ip].header2);i<=pcat_maxSize[ip];i++) strcat(headerStr2," ");
	  else if (tbForm==1)
	    for (i=strlen(defParam[ip].header2);i<=pcat_maxSize[ip]+4;i++) strcat(headerStr2," ");
	  if (pcat_errZero[ip]!=0 && tbForm==3)
	    {
	      strcat(headerStr,"        ");
	      strcat(headerStr2,"        ");
	    } 

	  /* Room for error */
	  if (pcat_errZero[ip]!=0 && (tbForm==0 || tbForm==4))
	    {
	      strcat(headerStr,"      ");
	      strcat(headerStr2,"      ");
	    }
	  if (pcat_errZero[ip]!=0 && tbForm==7)
	    {
	      strcat(headerStr,";");
	      strcat(headerStr2,";");
	    }
	  /* Room for reference */
	  if (pcat_refZero[ip]!=0 && (tbForm==0 || tbForm==4))
	    {
	      strcat(headerStr,"            ");
	      strcat(headerStr2,"            ");
	    }
	  if (pcat_refZero[ip]!=0 && tbForm==7)
	    {
	      strcat(headerStr,";");
	      strcat(headerStr2,";");
	    }
	}
      /* Add references for publication format */
      if (tbForm==1)
	{
	  for (j=0;j<np;j++)
	    {
	      ip = ptr[j];
	      if (pcat_refZero[ip]!=0)
		{
		  strcat(headerStr,defParam[ip].header1);
		  sprintf(dispStr,"%%%ds",10-strlen(defParam[ip].header1));
		  sprintf(temp,dispStr," ");
		  strcat(headerStr,temp);
		  strcat(headerStr2,"Ref.      ");
		}
	    }
	}
      if (tbForm!=5)
	{
	  if (tbForm!=6 && tbForm!=7)
	    {
	      for (i=strlen(headerStr);i--;)
		putc('-',stdout);
	      printf("\n");
	    }
	  puts(headerStr);
	  puts(headerStr2);
	  if (nohead==0 && tbForm!=6 && tbForm!=7)
	    {
	      for (i=strlen(headerStr);i--;)
		putc('-',stdout);
	      printf("\n");
	    }
	}
    }

  /* ************************** */
  /* Now display the main table */
  /* ************************** */
  for (i=0;i<npsr;i++)
    {
      if (tbForm!=5)
	{
	  if (noNumber==0)
	    {
	      if (tbForm==6 || tbForm==7)
		printf("%d;",i+1);
	      else
		printf("%-6d",i+1);
	    }
	}

      for (j=0;j<np;j++)
	{
	  dispParam = psrptr->param[ptr[j]];
	  sprintf(dispStr,"%%-%ds",pcat_maxSize[ptr[j]]);

	  /* long form => tbForm = 0 */
	  /* publish   => tbForm = 1 */
	  /* short     => tbForm = 2 */
	  /* short_error => tbForm = 3 */
	  /* short_error2 => tbForm = 4 */
	  	  
	  if (webflag==1 && ptr[j] == 2 && dispParam.set1==1 && (tbForm==0 || tbForm==1 || tbForm==4 || tbForm==6 || tbForm==7))
	    {
	      plusSign=0;
	      strcpy(temp,dispParam.val); /* Must replace + sign with %2B54 */
	      for (k=0;k<strlen(temp);k++)
		{
		  if (temp[k]=='+')
		    {
		      plusSign=1;
		      strcpy(temp2,temp+k);
		      temp[k]='\0';
		      strcat(temp,"%2B");
		      strcat(temp,temp2+1);
		    }		  
		}
	      if (tbForm==6 || tbForm==7)
		printf("%s;",dispParam.val);
	      else
		printf("<a href=\"%s?psr=%s\">%s</a>",PSRNAME_HTML,temp,dispParam.val);
	      if (tbForm==0) /* long form */
		{
		  for (k=strlen(dispParam.val);k<pcat_maxSize[ptr[j]];k++)
		    printf(" ");
		}
	    }
	  else if (webflag==1 && dispParam.set1==1 && (tbForm==0 || tbForm==1 || tbForm==4))
	    {
	      char temp2[1000]="",temp3[1000]="",temp4[1000]="";
	      char str[1000],str2[1000];
	      int pos=0,l,end=-1;
	      /* Replace anything within square brackets as a linked reference */
	      strcpy(temp,dispParam.val); 

	      for (k=0;k<strlen(temp);k++)
		{
		  if (temp[k]=='[' || end == 0)
		    {		      
		      int m;
		      char refTemp[100]="";
		      strcpy(refTemp,temp+pos);
		      refTemp[k-pos+1]='\0';
		      /* Note: was strcpy(temp2,refTemp) */
		      if (end==-1) strcat(temp2,refTemp);
		      /*		      printf("\ntemp2 = >%s<\n",temp2); */
		      for (m=k+1;m<strlen(temp);m++)
			{
			  if (temp[m]==',')
			    {
			      /*			      printf("In here %d<br>\n",m);*/
			      end=0;
			      break;
			    }
			  if (temp[m]==']')
			    {
			      if (end==0) end=1;
			      break;
			    }
			}
		      strcpy(refTemp,temp+k+1);
		      refTemp[m-k-1]='\0';
		      /*		      printf("\nrefTemp3 = >%s< %d\n",refTemp,end);*/
		      if (end==0) {sprintf(str,"<a href=\"http://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_ref.html#%s\">%s,</a>",refTemp,refTemp); m--;}
		      else if (end==1)
			{
			  sprintf(str,"<a href=\"http://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_ref.html#%s\">%s]</a>",refTemp,refTemp);
			  m--;
			}
		      else {sprintf(str,"<a href=\"http://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_ref.html#%s\">%s]</a>",refTemp,refTemp);}
		      strcat(temp2,str);
		      //strcat(temp4,temp2);
		      /*		      printf("This point >%s<\n",temp4); */
		      k=m;
		      pos=k+1;
		    }
		}
	      strcat(temp3,temp2);
	      if (strlen(temp3)==0) printf(dispStr,temp); 
	      else if (tbForm!=5)
		{
		  printf(dispStr,temp3); 
		  for (k=strlen(dispParam.val);k<pcat_maxSize[ptr[j]];k++)
		    printf(" ");
		}
	    }
	  else if (dispParam.set1==1 && (tbForm==0 || tbForm==4)) printf(dispStr,dispParam.val);
	  else if (dispParam.set1==1 && tbForm==7)
	    printf("%s;",dispParam.val);
	  else if (dispParam.set1==1 && tbForm==1) 
	    printf("%s",dispParam.val);
	  else if (dispParam.set1==1 && (tbForm==2 || tbForm==3 || tbForm==6)) 
	    {
	      /* Must check for string versus float value */
	      strcpy(temp,defParam[ptr[j]].shortStr);	      
	      if (tbForm==6)
		{
		  // Must remove the white space comment
		  if (strstr(temp,".")!=NULL)
		    {
		      char temp2[1024];
		      strcpy(temp2,"%");
		      strcat(temp2,strstr(temp,"."));
		      strcpy(temp,temp2);
		    }
		  strcat(temp,";");

		  if (strstr(temp,"s")==NULL)
		    printf(temp,dispParam.shortVal);
		  else
		    printf(temp,dispParam.val);
		}
	      else
		{
		  if (strstr(temp,"s")==NULL)
		    printf(defParam[ptr[j]].shortStr,dispParam.shortVal);
		  else
		    printf(defParam[ptr[j]].shortStr,dispParam.val);
		}
	    }
	  else if (tbForm==0 || tbForm==4) printf(dispStr,NO_VAL); 	  
	  else if (tbForm==7)
	    printf("%s;",NO_VAL);
	  else if (tbForm==1) {printf("%s",NO_VAL);strcpy(dispParam.val,NO_VAL);}
	  else if (tbForm==2 || tbForm==3 || tbForm==6) /* NULL string for SHORT */
	    {
	      if (tbForm==6)
		printf("%s;",NO_VAL);
	      else
		{
		  strcpy(temp,defParam[ptr[j]].shortStr);
		  strptr = strtok(temp,"%.-sgef");
		  strcpy(temp2,defParam[ptr[j]].shortStr);
		  if (strstr(temp2,"-")!=NULL)		
		    sprintf(temp,"%%-%ss",strptr);
		  else
		    sprintf(temp,"%%%ss",strptr);
		  printf(temp,NO_VAL);
		}
	    }
	  if (tbForm==0 || tbForm==3 || tbForm==4) putc(' ',stdout);
	  else if (tbForm==2) {putc(' ',stdout); putc(' ',stdout);}

	  /* ******************* */
	  /* Now print the error */
	  /* ******************* */
	  if (dispParam.set2==1 && tbForm==0) printf(" %4s ",dispParam.err);
	  else if (dispParam.set2==1 && tbForm==1)
	    {
	      printf("(%s)",dispParam.err);
	      /* Print a suitable number of spaces */
	      space = pcat_maxSize[ptr[j]]-(strlen(dispParam.val)+strlen(dispParam.err))+3;
	      sprintf(dispStr,"%%-%ds",space);
	      printf(dispStr," ");
	    }
	  else if (tbForm==1)
	    {
	      space = pcat_maxSize[ptr[j]]-strlen(dispParam.val)+5;
	      sprintf(dispStr,"%%-%ds",space);
	      printf(dispStr," ");
	    }
	  else if (tbForm==2 || tbForm==6)
	    {
	      /* Don't display errors in SHORT form */
	    }
	  else if ((tbForm==3 || tbForm==4 || tbForm==7) && pcat_errZero[ptr[j]]!=0)
	    {
	      if (dispParam.set2==1)
		{
		  if (tbForm==7)
		    {
		      sscanf(dispParam.err,"%d",&ierr);
		      printf("%.1e;",dispParam.error_expand*ierr);
		    }
		  else
		    {
		      sscanf(dispParam.err,"%d",&ierr);
		      printf("%.1e  ",dispParam.error_expand*ierr); 
		    }
		}
	      else
		{
		  if (tbForm==7)
		    printf("%s;",NO_ERR);
		  else
		    printf("      %s ",NO_ERR);
		}
	    }
	  else if (tbForm!=5 && pcat_errZero[ptr[j]]!=0) printf(" %4s ",NO_ERR);

	  if (dispParam.set3==1 && (tbForm==0 || tbForm==4 || tbForm==7)) 
	    { 
	      if (webflag==1 && tbForm!=7)
		{
		  printf(" <a href=\"http://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_ref.html#%s\">%s</a>&nbsp",dispParam.ref,dispParam.ref);
		  for (k=strlen(dispParam.ref);k<10;k++) /* add white space after reference */
		    printf(" ");
		}
	      else if (tbForm==7)
		printf("%s;",dispParam.ref);
	      else if (tbForm!=5)
		printf(" %-10s ",dispParam.ref);
	    }
	  else if (tbForm!=2 && tbForm!=6 && tbForm!=3 && tbForm!=1 && tbForm!=5 && pcat_refZero[ptr[j]]!=0)
	    {
	      if (tbForm==7) 
		printf("%s;",NO_REF);
	      else
		printf(" %-10s ",NO_REF);
	    }
	}
      /* For publication output now put the references */
      if (tbForm==1)
	{
	  for (j=0;j<np;j++)
	    {
	      dispParam = psrptr->param[ptr[j]];
	      if (dispParam.set3==1)
		{
		  if (webflag==1)
		    {
		      printf(" <a href=\"http://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_ref.html#%s\">%7s</a>  ",dispParam.ref,dispParam.ref);
		    }
		  else
		    printf(" %7s  ",dispParam.ref);
		}
	      else if (pcat_refZero[ptr[j]]!=0) printf("[%7s] ",NO_REF);
	    }
	}
      if (tbForm!=5)
	{
	  puts("");
	  if ((i+1)%5==0 && nohead==0 && noNumber==0 && i!=npsr-1 && tbForm!=6 && tbForm!=7)puts("");
	}
      psrptr = (list[i].post)->current; 
    }
  if (tbForm!=5 && tbForm!=6 && tbForm!=7) 
    {
      for (i=strlen(headerStr);i--;)
	putc('-',stdout);
      if (nohead==0)
	printf("\n");
    }
}
