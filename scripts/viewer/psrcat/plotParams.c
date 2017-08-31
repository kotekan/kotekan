#include <stdio.h>
#include <string.h>
#include "psrcat.h"
#include <math.h>

int calc_tick(double x1,double x2, double *x1i,double *x2i, double *xi);
int plimit(double x1, double x2, double *xi, double *x1i, double *x2i);
int nint(double x);

/* Function to create plotting instructions for the web-based JAVA plotting routines */
void plotParams(char *plotx,char *ploty,int plotx_log,int ploty_log,pulsar *psr,
		linkedList *list,int npsr)
{
  int i,count,ignorepsr,badval,ip,ipb;
  double xval[MAX_PSR],yval[MAX_PSR];
  double xmin=0,xmax=0,ymin=0,ymax=0,xmin1,xmax1,xstep,ymin1,ymax1,ystep;
  char psrname[MAX_PSR][MAX_STRLEN],teststr[MAX_STRLEN],temp[MAX_STRLEN];
  char plotx2[1000],ploty2[1000];
  int psrtype[MAX_PSR];
  int merged_font = 5;
  
  /* Find minimum and maximum values */
  
  /* See how many pulsars we are merging to choose a suitable plot symbol */
  /* If less than 5 then use a large symbol, otherwise use a small symbol */
  count=0;  
  for (i=0;i<npsr;i++)
    {
      if (psr[i].merged == 1)
	count++;
    }
  if (count>5)
    merged_font = 6;
  count=0; 
  sprintf(temp,"result = %s",plotx);  strcpy(plotx2,temp);
  sprintf(temp,"result = %s",ploty);  strcpy(ploty2,temp);

  for (i=0;i<npsr;i++)
    {
      badval=0;
      sprintf(temp,"result = %s",plotx);  strcpy(plotx2,temp);
      sprintf(temp,"result = %s",ploty);  strcpy(ploty2,temp);
      
      /* Evalulate the expressions for the x-axis and for the y-axis */
      xval[count] = formExpression(plotx2,&psr[i],&badval,0);
      if (badval==1)
	ignorepsr = 1;
      else
	{
	  if (strlen(ploty2)>9) /* Is there anything to plot on y-axis */
	    yval[count] = formExpression(ploty2,&psr[i],&badval,0);
	  else
	    yval[count] = 1.0;
	  if (badval!=1)
	    {
	      ignorepsr = 0; 
	      ip = getParam("NAME",&psr[i]);
	      strcpy(psrname[count],psr[i].param[ip].val);
	  
	      psrtype[count] = 1;
	      ipb = getParam("BINARY",NULL);      
	      ip = getParam("TYPE",NULL);
	      if (psr[i].param[ipb].set1==1)
		psrtype[count] = 2;
	      if (psr[i].param[ip].set1==1)
		{
		  if (strstr(psr[i].param[ip].val,"HE")!=NULL)
		      //		      || strcmp(psr[i].param[ip].val,"NR")==0)
		    psrtype[count] = 4;
		  if (strstr(psr[i].param[ip].val,"AXP")!=NULL)
		    psrtype[count] = 3;
		}
	      if (psr[i].merged == 1)
		psrtype[count] = merged_font;
	      if ((plotx_log == 1 && xval[count]<=0.0) || 
		  (ploty_log == 1 && yval[count]<=0.0) || ignorepsr == 1)
		{
		}
	      else
		{
		  if (count==0)
		    {
		      xmin = xval[count];
		      xmax = xval[count];
		      ymin = yval[count];
		      ymax = yval[count];
		    }
		  else
		    {
		      if (xval[count] > xmax) xmax=xval[count];
		      if (yval[count] > ymax) ymax=yval[count];
		      if (xval[count] < xmin) xmin=xval[count];
		      if (yval[count] < ymin) ymin=yval[count];
		    }
		  count++;
		}
	      /* output PTplot FORMAT         */
	      /* printf("%f %f\n",xval,yval); */
	    }
	}
    }
  if (plotx_log == 1) /* Update log scale */
	{
	  xmin1 = ((int)log10(xmin))-1.0;
	  xmax1 = ((int)log10(xmax))+1.0;
	  xstep = 1.0;
	}
      else
	{
	  calc_tick(xmin,xmax,&xmin1,&xmax1,&xstep);    
	  if ((xmax1-xmin1)/(xstep)>20)
	    xstep*=2.0;
	}
      if (ploty_log == 1) /* Update log scale */
	{
	  ymin1 = ((int)log10(ymin)-1.0);
	  ymax1 = ((int)log10(ymax)+1.0);
	  ystep = 1.0;
	}
      else
	{
	  if (strlen(ploty2)>0)
	    {
	      calc_tick(ymin,ymax,&ymin1,&ymax1,&ystep);
	      if ((ymax1-ymin1)/(ystep)>20)
		ystep*=2.0;
	    }
	  else
	    {
	      ymin1 = 1;
	      ymax1 = 1;
	      ystep =1;
	    }
	}
      for (i=0;i<count;i++)
	{
	  if (plotx_log == 1) xval[i] = log10(xval[i]);
	  if (ploty_log == 1) yval[i] = log10(yval[i]);
	}
      printf("<applet code=\"PlotApplet\" width=850 height=800>\n");
      if (strlen(ploty2)<10)
	printf("<param name=\"plottype\" value=\"histogram\">\n");
      else
	printf("<param name=\"plottype\" value=\"xyplot\">\n");
      printf("<param name=\"xmin\" value=\"%e\">\n",xmin1);
      printf("<param name=\"xmax\" value=\"%e\">\n",xmax1);
      printf("<param name=\"xstep\" value=\"%e\">\n",xstep);
      printf("<param name=\"ymin\" value=\"%e\">\n",ymin1);
      printf("<param name=\"ymax\" value=\"%e\">\n",ymax1);
      printf("<param name=\"ystep\" value=\"%e\">\n",ystep);
      printf("<param name=\"title\" value=\" \">\n");
      printf("<param name=\"xlabel\" value=\"%s\">\n",plotx2+9);
      printf("<param name=\"ylabel\" value=\"%s\">\n",ploty2+9);
      printf("<param name=\"xlog\" value=\"%d\">\n",plotx_log);
      printf("<param name=\"ylog\" value=\"%d\">\n",ploty_log);
      
      for (i=0;i<count;i++)
	{
	  /* New format */
	  printf("<param name=\"x%d\" value=\"%e\">\n",i+1,xval[i]);
	  printf("<param name=\"y%d\" value=\"%e\">\n",i+1,yval[i]);
	  printf("<param name=\"name%d\" value=\"%s\">\n",i+1,psrname[i]);
	  printf("<param name=\"type%d\" value=\"%d\">\n",i+1,psrtype[i]);
	}
      printf("</applet>\n");
      /*printf("</html>\n"); */

}


int calc_tick(double x1,double x2, double *x1i,double *x2i, double *xi)
{
  if(plimit(x1,x2,xi,x1i,x2i)!=0)
      printf("Warning in making plots: x2 !> x1 !!\n");
  return 0;
}

int plimit(double x1, double x2, double *xi, double *x1i, double *x2i){

  /* Limits and major tick interval for plot axis */
  /* Returns 1 if x1>=x2, 0 if OK */
  /* RNM 16 Jan 03 */

  int nlxi;
  double xr,axi;
    
  xr=x2-x1;
  if(xr<=0.){
    return(1);
  }
  axi=log10(xr/10.);
  nlxi=nint(axi);
  axi=pow(10.,(double)nlxi);

  *xi=axi;
  *x1i=(int)(x1/axi-1.)*axi;
  *x2i=(int)(x2/axi+1.)*axi;

  return(0);
}

int nint(double x){
  int i;
  if(x>0.){
    i=(int)(x+0.5);
  }
  else{
    i=(int)(x-0.5);
  }
  return(i);
}

