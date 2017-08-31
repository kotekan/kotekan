#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "psrcat.h"
 
pulsar psr[MAX_PSR]; 

char pcat_parameters[MAX_PARAM][20];  
paramtype defParam[MAX_PARAM];
  
float recordCompare(void const *a,void const *b);
void displayEphemeris(pulsar *psr,linkedList *list,int npsr,int *ptr,int np,paramtype *defParam,int ephem,int webflag);
void mySort(pulsar *psr,linkedList *list,int npsr);
void get_bibliography(pulsar *psr,int numpsr);
void listReferences(int flag,char *ref,pulsar *psr,int npsr,int *ptr,int np);

int main(int argc,char *argv[])
{
  int webflag = 0;  /* Output in html format? */
  int npsr=0,allmerge=0,bib_info=0;
  register unsigned int i,j;
  int nocand=0,nointerim=0;
  int np=3,ptr[MAX_PLIST]={0,9,7};
  char sptr[MAX_PLIST][MAX_STRLEN];
  linkedList list[MAX_PSR];
  int iname,ephem=0,listref=0,listallref=0,checkCat=0;
  int haveName[MAX_PSR];
  char plotx[MAX_STRLEN],ploty[MAX_STRLEN];
  char selectNames[MAX_PSR][MAX_NAMELEN];
  char filterStr[MAX_FILTERSTR]="result = ";
  int exactMatch=0;
  char dbFile[MAX_CATLEN]="";
  char boundary[MAX_STRLEN]="";
  char mergeCat[MAX_MERGELEN]="";
  char groupStr[MAX_MERGELEN],tempMerge[MAX_MERGELEN];
  char listIndRef[MAX_STRLEN];
  char *customStr,*strptr,*savePtr,mergeit;
  int fsize=3;
  int plotx_log,ploty_log;
  int listparticularRef=0;
  int nohead=0;
  int noNumber=0,nread;
  int tbForm = 0; /* Long format */
  FILE *groupFile;
  int aliasCount=0;
  int debug=0;
  char alias1[MAX_ALIASES][MAX_STRLEN],alias2[MAX_ALIASES][MAX_STRLEN];
  if (debug==1) printf("Starting off\n");
  defineParameters(defParam);
  if (debug==1) printf("defined parameters\n");
  /* Initialise */
  pcat_sort=-1;
  pcat_descend=0;
  verbose=0;
  strcpy(NO_VAL,"*");
  strcpy(pcat_custom1,"result = ");
  strcpy(pcat_custom2,"result = ");
  strcpy(pcat_custom3,"result = ");
  strcpy(pcat_custom4,"result = ");
  strcpy(plotx,""); strcpy(ploty,""); plotx_log = 0; ploty_log = 0;
  for (j=0;j<MAX_PSR;j++)
    {
      for (i=0;i<MAX_PARAM;i++)
	psr[j].param[i].derived = 0;
    }

  parseParameters(argc,argv,ptr,sptr,&np,selectNames,&iname,haveName,
		  filterStr,customStr,&nohead,&tbForm,
		  &noNumber,dbFile,mergeCat,defParam,&ephem,boundary,&allmerge,
		  &webflag,plotx,ploty,&plotx_log,&ploty_log,&bib_info,&listref,
		  &listallref,&listparticularRef,listIndRef,&checkCat,&fsize,&nocand,
		  &nointerim,&exactMatch);
  if (debug==1) printf("parsed parameters\n");
  if (nocand==1)
    {
      if (strlen(filterStr) > 9)
	strcat(filterStr," && !TYPE(CAND) ");
      else
	strcat(filterStr," !TYPE(CAND) ");
    }
  if (nointerim==1)
    {
      if (strlen(filterStr) > 9)
	strcat(filterStr," && !TYPE(INTERIM) ");
      else
	strcat(filterStr," !TYPE(INTERIM) ");
    }
  if (allmerge == 1) 
    {
      strcat(mergeCat," ");
      /*      mergeit = getenv("PSRCAT_RUNDIR");*/
      if (getenv("PSRCAT_RUNDIR")==NULL)
	printf("Warning PSRCAT_RUNDIR not set\n");
      strcat(mergeCat,getenv("PSRCAT_RUNDIR"));
      strcat(mergeCat,"/obs*.db");
    }

  /* Check correctly for aliases */
  if (debug==1) printf("looking to merge catalogues\n");
  if (strlen(mergeCat)>0)
    {
      char sysstr[1000];
      char dummy[1000];
     FILE *pin;
     /* ISSUE: THESE LINES SHOULD BE SWITCHED FOR SOLARIS AND FOR LINUX */

     /* The egrep line is for solaris */
     if (webflag==0)       
         sprintf(sysstr,"sed -n -e '/psrj\\|alias\\|PSRJ\\|ALIAS/p' %s | grep -B 1 -i alias | awk '{print $2}'",mergeCat);   
       /*                   sprintf(sysstr,"/usr/local/gnu/bin/egrep -i \"psrj|alias\" %s | /usr/local/gnu/bin/grep -B 1 -i alias | awk '{print $2}'",mergeCat); */
     else 
       sprintf(sysstr,"/usr/local/gnu/bin/egrep -i \"psrj|alias\" %s | /usr/local/gnu/bin/grep -B 1 -i alias | awk '{print $2}'",mergeCat); 
       //       sprintf(sysstr,"/bin/egrep -i \"psrj|alias\" %s | /usr/local/gnu/bin/grep -B 1 -i alias | awk '{print $2}'",mergeCat); 
       
     
     if (!(pin = popen(sysstr,"r")))
       printf("Unable to run %s\n",sysstr);

      while (!feof(pin))
	{
	  if (fscanf(pin,"%s %s",alias1[aliasCount],alias2[aliasCount])==2)
	    {
	      aliasCount++; 
	      if (aliasCount > MAX_ALIASES)
		{
		  printf("ERROR: too many aliases\n");
		  exit(1);
		}
	    }
	}
      pclose(pin);
    }


  for (i=0;i<MAX_PARAM;i++)
    {
      if (strlen(defParam[i].header1)>=strlen(defParam[i].header2))
	pcat_maxSize[i]=strlen(defParam[i].header1);
      else
	pcat_maxSize[i]=strlen(defParam[i].header2);
      pcat_refZero[i]=0;
      pcat_errZero[i]=0;
    }
  if (debug==1) printf("Reading catalogues\n");
  if (readCatalogue(psr,list,&npsr,selectNames,iname,haveName,filterStr,ptr,sptr,np,dbFile,0,strlen(mergeCat),boundary,defParam,checkCat,alias1,alias2,aliasCount,exactMatch)==1)
    exit(1); /* Cannot open file */

  if (debug==1) printf("Considering to merge catalogues\n");
  /* **************************** */
  /* Merge catalogues if required */
  /* **************************** */
  if (strlen(mergeCat)>0)
    {
      strptr = strtok_r(mergeCat," ;,\t\n",&savePtr);	  
      while (1)
	{
	  /* ****************************** */
	  /* Check if this is a group merge */
	  /* ****************************** */
	  sprintf(groupStr,"ls -1 %s",strptr);
	  groupFile = popen(groupStr,"r");
	  while (!feof(groupFile))
	    {
	      nread = fscanf(groupFile,"%s",tempMerge);

	      if (nread==1) readCatalogue(psr,list,&npsr,selectNames,iname,haveName,filterStr,ptr,sptr,np,tempMerge,1,1,boundary,defParam,checkCat,alias1,alias2,aliasCount,exactMatch);      	  
	    }
	  pclose(groupFile);
	  strptr = strtok_r(NULL," ;,\t\n",&savePtr);
	  if (strptr==NULL) break;
	}
      /* Must now sort the catalogue */
      if (pcat_sort==-1) pcat_sort=0;
    }
  if (checkCat==1)
    exit(1);

  /* If we have merged catalogues, we must now do the filtering */
  if (strlen(mergeCat)>0)
    {
      int filter=1;
      int badval,sub=0,count=0,j;
      linkedList temp[MAX_PSR];
      pulsar *check;

      for (i=0;i<npsr;i++)
	temp[i] = list[i];

      check =  temp[0].current;
      for (i=0;i<npsr;i++)
	{
	  if (strlen(filterStr)<10 || formExpression(filterStr,check,&badval,exactMatch)==1) /* Don't filter */
	    {
	      list[count].current = check;
	      if (count!=0)
		list[count-1].post = &(list[count]);
	      count++;
	      /* Check that we have all the parameters */
	      for (j=np;j--;)
		{
		  if (check->param[ptr[j]].set1 != 1)
		    deriveParameter(sptr[j],check,ptr[j]);
		} 
	    }
	  check = (temp[i].post)->current;
	}
      npsr = count;
    }
  

  /* Check that all pulsar names given on the command line exist */
  for (i=0;i<iname;i++)
    {
      if (haveName[i]!=1)
	{
	  if (webflag==0)
	    printf("WARNING: PSR %s not in catalogue\n",selectNames[i]);
	  else 
	    printf("<pre>WARNING: PSR %s not in catalogue</pre> \n",selectNames[i]); 
	}
    }
  /* Sort the catalogue */
  if (pcat_sort!=-1) mySort(psr,list,npsr);   
  if (bib_info==1)
    {
      get_bibliography(psr,npsr);
      return 0;
    }
  /* Set up display for web */
  if (webflag==1)
    {
      char str[1000];
      sprintf(str,"<font face = \"Courier\" size=%d>",fsize);
      printf("%s\n",str);
      printf("<pre>\n");
    }
  if (strlen(plotx)>0)
    plotParams(plotx,ploty,plotx_log,ploty_log,psr,list,npsr);
  else
    {
      if (ephem==0)
	{
	  if (listparticularRef==0 && listallref==0) displayOutput(psr,list,npsr,ptr,np,defParam,nohead,tbForm,noNumber,webflag);
	  if (listref==1) /* Now consider references */
	    listReferences(1,"",psr,npsr,ptr,np);
	  else if (listref==2)
	    listReferences(4,"",psr,npsr,ptr,np);
	  else if (listref==3)
	    listReferences(5,"",psr,npsr,ptr,np);
	  else if (listallref==1) /* List complete bibliographic reference file */
	    listReferences(2,"",psr,npsr,ptr,np);
	  else if (listparticularRef==1) /* List individual reference */
	    listReferences(3,listIndRef,psr,npsr,ptr,np);
	}
      else
	displayEphemeris(psr,list,npsr,ptr,np,defParam,ephem,webflag);
    }
  /* Finish off */
  if (webflag == 1)
    {   
      printf("</pre>\n");
      printf("</font>\n");
    } 
  return 0;
}

void displayEphemeris(pulsar *psr,linkedList *list,int npsr,int *ptr,int np,paramtype *defParam,int ephem,int webflag)
{
  int i,j,ierr,ip,ip2,ip3,ip4,ip5,ip6,ip7,ip8,ip9,eph,if0,if1,ip0,ip1;
  char str[1000];
  pulsar *psrptr;
  parameter dispParam;

  psrptr = list[0].current;

  ip2=getParam("ELONG",NULL);
  ip3=getParam("ELAT",NULL);
  ip4=getParam("PMRA",NULL);
  ip5=getParam("PMDEC",NULL);
  ip6=getParam("RAJ",NULL);
  ip7=getParam("DECJ",NULL);
  ip8=getParam("PMELONG",NULL);
  ip9=getParam("PMELAT",NULL);
  if0=getParam("F0",NULL);
  if1=getParam("F1",NULL);
  ip0=getParam("P0",NULL);
  ip1=getParam("P1",NULL);

  for (i=0;i<npsr;i++)
    {
      if (i!=0)
	printf("@-----------------------------------------------------------------\n");
      if (ephem==2 || ephem==1 || ephem==4)
	{
	  for (j=0;j<MAX_PARAM;j++)
	    {
	      if (ephem==2 || ephem==4 || defParam[j].ephem==1)
		{
		  if (strcmp(pcat_parameters[j],"C1")!=0 &&
		      strcmp(pcat_parameters[j],"C2")!=0 &&
		      strcmp(pcat_parameters[j],"C3")!=0 &&
		      strcmp(pcat_parameters[j],"C4")!=0)
		    {
		      ip=getParam(pcat_parameters[j],psrptr);
		      dispParam = psrptr->param[j];
		    }
		}
	      if (ephem==2 || ephem==4 || (defParam[j].ephem==1 && ((dispParam.derived!=1 || j==if0 || j==if1) && j!=ip0 && j!=ip1)))
		{
		  //		  printf("j = %d\n",j);
		  if (strcmp(pcat_parameters[j],"C1")!=0 &&
		      strcmp(pcat_parameters[j],"C2")!=0 &&
		      strcmp(pcat_parameters[j],"C3")!=0 &&
		      strcmp(pcat_parameters[j],"C4")!=0)
		    {
		      /* Output raj and decj if have elat and elong, but pmra and pmdec */
		      /*		      if (ip==ip2 && (psrptr->param[ip4].set1==1 
				      || psrptr->param[ip5].set2==1))
			{
			  ip = ip6;
			  dispParam = psrptr->param[ip];
			  strcpy(pcat_parameters[j],"RAJ");
			}
		      if (ip==ip3 && (psrptr->param[ip4].set1==1 
				      || psrptr->param[ip5].set2==1))
			{
			  ip = ip7;
			  dispParam = psrptr->param[ip];
			  strcpy(pcat_parameters[j],"DECJ");
			  } */
		      if (dispParam.set1==1) {
			printf("%-15.15s %-25.25s",pcat_parameters[j],dispParam.val);
			if (dispParam.set2==1)
			  {			    
			    sscanf(dispParam.err,"%d",&ierr);
			    sprintf(str,"%.3e",dispParam.error_expand*ierr);
			    printf("%-10.10s",str);
			    if (ephem==4)
			      printf("[%s] ",dispParam.ref);
			  }
			else if (ephem==4)
			  printf("%-10.10s[%s] ","",dispParam.ref);
			printf("\n");
		      }
		    }
		}
	    }
	}
      else if (ephem==3)
	{
	  for (j=0;j<np;j++)
	    {
	      dispParam = psrptr->param[ptr[j]];
	      if (dispParam.set1==1) {
		printf("%-15.15s %-25.25s",pcat_parameters[ptr[j]],dispParam.val);
		if (dispParam.set2==1)
		  {
		    sscanf(dispParam.err,"%d",&ierr);
		    printf("%.3e",dispParam.error_expand*ierr);
		  }
		printf("\n");
	      }
	    }
	}
      // Now setting EPHVER in readCatalogue
      //      eph = getParam("EPHVER",NULL);
      //      if (psrptr->param[eph].set1==1)
      //	printf("%-15.15s\t%-25.25s\n","EPHVER",psrptr->param[eph].val);

      psrptr = (list[i].post)->current;
    }
}


/* Have got a linked list that requires sorting, the parameter to sort could be */
/* alphabetic or numeric                                                        */

/* At the moment just doing a poor bubble sort ... */

void mySort(pulsar *psr,linkedList *list,int npsr)
{
  register unsigned int i;
  float ret;
  register unsigned int change;
  pulsar *psr1,*psr2;
  pulsar *temp1;
  linkedList *temp2;

  do {
    change=0;
    for (i=1;i<npsr;i++)
      {
	if ((ret = recordCompare(list[i-1].current,list[i].current))>0)
	  {
	    temp1 = list[i-1].current;
	    temp2 = list[i-1].post;

	    list[i-1].current = list[i].current;

	    list[i].current   = temp1;
	    change=1;
	  }	
	psr1 = psr2;
      }
  } while (change==1);
}

float recordCompare(void const *a,void const *b)
{
  register unsigned int s1 = ((pulsar *)a)->param[pcat_sort].set1;
  register unsigned int s2 = ((pulsar *)b)->param[pcat_sort].set1;

  if (s1 == 0) /* Maybe need to derive this variable */
    {
      getParam(pcat_parameters[pcat_sort],
	       ((pulsar *)a));
      s1 = ((pulsar *)a)->param[pcat_sort].set1;
    }
  if (s2 == 0) /* Maybe need to derive this variable */
    {
      getParam(pcat_parameters[pcat_sort],
	       ((pulsar *)b));
      s2 = ((pulsar *)b)->param[pcat_sort].set1;
    }


  if (s1 != 1 && s2 != 1)
    return 0; 
  if (s1 != 1 && s2 == 1)
    return -1; 
  if (s1 == 1 && s2 != 1)
    return 1; 

  if (pcat_descend==1) /* Sort in descending order */
    {
      if (defParam[pcat_sort].numeric==1)
	return ((pulsar *)b)->param[pcat_sort].shortVal-((pulsar *)a)->param[pcat_sort].shortVal;
      else
	return strcmp(((pulsar *)b)->param[pcat_sort].val,((pulsar *)a)->param[pcat_sort].val);
    }
  else
    {
      if (defParam[pcat_sort].numeric==1)
	return ((pulsar *)a)->param[pcat_sort].shortVal-((pulsar *)b)->param[pcat_sort].shortVal;
      else
	return strcmp(((pulsar *)a)->param[pcat_sort].val,((pulsar *)b)->param[pcat_sort].val);
  }
}



/* **************************************************** */
/* get_bibliography                                     */
/* Author:  G. Hobbs                                    */
/* Date:    22 Dec 2002 (ATNF)                          */
/*                                                      */
/* Purpose: produces a list of the references used in   */
/* the catalogue                                        */
/*                                                      */
/* Note: associations can have references built in      */
/* using [ref].                                         */
/* **************************************************** */

void get_bibliography(pulsar *psr,int numpsr)
{
  int i,j,k,start=0;
  char reference[MAX_STRLEN];

  printf("\\documentclass{article}\n");
  printf("\\begin{document}\n");
  for (i=0;i<numpsr;i++)
    {
      for (j=0;j<MAX_PARAM;j++)
        {
          if (strcmp(psr[i].param[j].ref,"*")!=0 && strlen(psr[i].param[j].ref) > 0)
            printf("\\cite{%s}\n",psr[i].param[j].ref);
	  /* Must look for references in the associations line */
	  if (j==getParam("ASSOC",NULL) || j==getParam("TYPE",NULL) || j==getParam("BINCOMP",NULL))
	    {
	      start=-1;
	      for (k=0;k<strlen(psr[i].param[j].val);k++)
		{
		  if (psr[i].param[j].val[k]=='[') /* Reference */
		      start=k;
		  else if (psr[i].param[j].val[k]==',' && start!=-1)
		    {
		      strncpy(reference,psr[i].param[j].val+start+1,k-start);
		      reference[k-start-1]='\0';
		      printf("\\cite{%s}\n",reference);
		      start = k;
		    }
		  else if (psr[i].param[j].val[k]==']')
		    {
		      strncpy(reference,psr[i].param[j].val+start+1,k-start);
		      reference[k-start-1]='\0';
		      start=-1;
		      printf("\\cite{%s}\n",reference);
		    }
		}
	    }
        }
    }

  printf("\\bibliographystyle{mylistall}\n");
  printf("\\bibliography{journals,modrefs,psrrefs,crossrefs}\n");
  printf("\\end{document}\n");
}

/* List bibliographic references */
/* Produce file using (on the webpage .html file) and delete header information:
sed s/"<tr>"/""/g psrcat_ref | sed s/"<\/a>"/" "/g | sed s/"<b>"/" "/g | sed s/"<td valign=top>"/""/g | sed s/"<\/b>"/" "/g | sed s/"<\/td>"/" "/g | sed s/"<td>"/" "/g | sed s/"<\/tr>"/""/g | sed s/" :"/":"/g | sed s/"<i>"/" "/g | sed s/"<\/i>"/""/g | sed s/"<a NAME="/"***"/g | sed s/">"/""/g
*/

void listReferences(int flag,char *giveref,pulsar *psr,int npsr,int *ptr,int np)
{
  FILE *fin,*pin,*fout;
  char fname[100],first[100];
  char str[1000],ref[100];
  char listRef[1000][1000];
  int countRef=0,found=0;
  int  i,j,k;

  //  printf("In here with flag: %d, np = %d\n",flag,np);
  
  if (flag==4)
    {
      fout = fopen("dummy.tex","w");
      fprintf(fout,"\\documentclass{article}\n");
      fprintf(fout,"\\begin{document}\n");
    }

  if (flag==1 || flag==4 || flag == 5)
    {
      for (i=0;i<npsr;i++)
	{
	  for (j=0;j<np;j++)
	    {
	      //	      printf("Comparing >%s< %d\n",psr[i].param[ptr[j]].ref,psr[i].param[ptr[j]].set3);
	      //	      if (strcmp(psr[i].param[ptr[j]].ref,"*")!=0 && strlen(psr[i].param[ptr[j]].ref) > 0)
	      if (psr[i].param[ptr[j]].set3 == 1)
		{
		  found=0;
		  for (k=0;k<countRef;k++)
		    {
		      if (strcmp(psr[i].param[ptr[j]].ref,listRef[k])==0)
			{
			  found=1;
			  break;
			}
		    }
		  if (found==0)
		    {
		      strcpy(listRef[countRef],psr[i].param[ptr[j]].ref);
		      if (flag==5) printf("%s ",listRef[countRef]);
		      countRef++;
		    }
		}
	    }
	}
    }
  //  printf("Number of references = %d\n",countRef);
  if (getenv("PSRCAT_RUNDIR")==NULL)
    {
      printf("Warning PSRCAT_RUNDIR not set\n");      
      exit(1);
    }
  strcpy(fname,getenv("PSRCAT_RUNDIR"));
  strcat(fname,"/psrcat_ref");


  if (!(fin = fopen(fname,"r")))
    {
      printf("Unable to open filename >%s<\n",fname);
      exit(1);
    }
  found=0;
  while (!feof(fin))
    {
      if (fgets(str,1000,fin)!=NULL)
	{
	  str[strlen(str)-1]='\0';

	  sscanf(str,"%s",first);
	  if (strstr(first,"***")!=NULL)
	    {
	      found=0;
	      strcpy(ref,first+3);
	      if (flag==1 || flag==4)
		{
		  for (k=0;k<countRef;k++)
		    {
		      if (strcmp(listRef[k],ref)==0)
			{
			  found=1;
			  if (flag==4)
			    fprintf(fout,"\\cite{%s}\n",first+3);
			  else
			    printf("\n\n%s",str+strlen(first)+2);
			}
		    }
		}
	      if (flag==2) printf("\n\n%s",str+strlen(first)+2);
	      if (flag==3 && strcasecmp(ref,giveref)==0) printf("\n%s",str+strlen(first)+2);
	    }
	  else if (flag==1 && found==1)
	    printf("%s",str);
	  else if (flag==2)
	    printf("%s",str);
	  else if (flag==3  && strcasecmp(ref,giveref)==0)
	    printf("%s",str);
 
	}
    }
  if (flag==4)
    {
      FILE *fin2;
      char strVal[1000];
      fprintf(fout,"\\bibliography{modrefs,psrrefs,crossrefs}\n");
      fprintf(fout,"\\bibliographystyle{mn}\n");
      fprintf(fout,"\\end{document}\n");
      fclose(fout);
      system("\\rm dummy.bbl");
      system("latex dummy.tex > dummy.out");
      system("bibtex dummy >> dummy.out");
      printf("<pre>\n");
      fin2 = fopen("dummy.bbl","r");
      while (!(feof(fin2)))
	{
	  if (fgets(strVal,1000,fin2)!=NULL)
	    printf("%s",strVal);
  
	}
	     

      printf("</pre>\n");
    }
  printf("\n\n");
  fclose(fin);
}
