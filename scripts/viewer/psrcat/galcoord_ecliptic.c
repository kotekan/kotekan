/* Converts to gl and gb based on hla93 (enter proper motion in ecliptic coords) */

#include <stdio.h>
#include <string.h>
#include <math.h>

void ec2gal_ecliptic(double dr,double dd,double *dl,double *db,double rotmat[3][3]);
void transform_ecliptic(double dm[4][4],double *v1,double *v2);
void dir2sph_ecliptic(double *v,double *a, double *b);
double f_mod_ecliptic(double a,double p);
double product_ecliptic(double *va,double *vb);
double f_sign_ecliptic(double a,double b);
void matx2v_ecliptic(double mat[3][3],double *v,double *r);
void multiply_ecliptic(double a[3][3],double b[3][3],double c[3][3]);

 
/* Takes el, eb (rad), pmelong, pmelong_error, pmelat,pmelat_error (mas/yr), */
/* dist (kpc) and returns proper motion in Galactic coordinates with errors   */
int find_galcoord_ecliptic(double el,double eb,double pmel,double pmelerr,double pmeb,double pmeberr, double dist,double *mul,double *mulerr,double *mub,double *muberr)
{
  double l2000,b2000,cosb;
  double cel2gal[3][3],cel[3],gal[3],covmat[3][3];
  double covar=0.0,covmatgal[3][3],covargal;
  double galcart[4],gcr,ddtemp,pmlerr,pmberr;
  double p[4],q[4],r[4],vsl,vsb,vgal[4];
  double pml,pmb,thetagal,vrgal,pmlrot,pmbrot;
  int i;
  
  /* angles in mas/yr to kms/s (along with distances in kpc) */
  double dang2vel = 4.7405;
  /* Oort's consts */
  double a=14.5,b=-12.0,r0=8.5,v0gal,vsun[4];
  /* Solar velocity */
  vsun[1]=9.2; vsun[2]=10.5; vsun[3]=6.9;
  /* Galactic rotation velocty at the Earth */
  v0gal = 225.0; /* Using flat rotation curve */

  /* Calculate L and B */
  ec2gal_ecliptic(el,eb,&l2000,&b2000,cel2gal);
  cel[1]=pmel;
  cel[2]=pmeb;

  /* Calculate the position in galactic cartesian coordinates */
  /* x increasing towards the galactic centre */
  cosb=cos(b2000);
  galcart[1] = cos(l2000)*cosb;
  galcart[2] = sin(l2000)*cosb;
  galcart[3] = sin(b2000);

  for (i=1;i<=3;i++)
    galcart[i]=galcart[i]*dist; 

  /* Calculate the projected galactocentric distance to the source */
  gcr=sqrt(pow(r0-galcart[1],2)+pow(galcart[2],2)); 
  /* Calculate the proper motions in galactic coords */
  matx2v_ecliptic(cel2gal,cel,gal);

  /* Form the covarience matrix */
  covmat[1][1] = pmelerr*pmelerr;
  covmat[2][2] = pmeberr*pmeberr;
  covmat[1][2] = covar*pmelerr*pmeberr;
  covmat[2][1] = covar*pmelerr*pmeberr;

  /* Postmultiply by the conversion matrix */
  multiply_ecliptic(covmat,cel2gal,covmatgal);

  /* Premultiply by the transpose of the conversion matrix */
  ddtemp = cel2gal[1][2];
  cel2gal[1][2] = cel2gal[2][1];
  cel2gal[2][1]=ddtemp;
  multiply_ecliptic(cel2gal,covmatgal,covmatgal);

  pmlerr = sqrt(fabs(covmatgal[1][1]));
  pmberr = sqrt(fabs(covmatgal[2][2]));
  covargal = covmatgal[1][2]/fabs(pmlerr)/fabs(pmeberr);

  /* Calculate orthogonal vectors */
  p[1] = -sin(l2000);
  p[2] =  cos(l2000);
  p[3] =  0.0;

  q[1] = -sin(b2000)*cos(l2000);
  q[2] = -sin(b2000)*sin(l2000);
  q[3] =  cos(b2000);

  r[1] =  cos(b2000)*cos(l2000);
  r[2] =  cos(b2000)*sin(l2000);
  r[3] =  sin(b2000);

  /* Form dot product with the sun's velocity vector */
  vsl = -product_ecliptic(p,vsun)/dist/dang2vel;
  vsb = -product_ecliptic(q,vsun)/dist/dang2vel; 

  /* Find the proper motions expected from galactic rotation using rotation */
  /* curve model */
  thetagal = atan2(galcart[2],r0-galcart[1]);
  vrgal = v0gal; /* Use flat rotation curve */
  vgal[1] = vrgal*sin(thetagal);
  vgal[2] = vrgal*cos(thetagal)-v0gal;
  vgal[3] = 0.0;

  pmlrot=product_ecliptic(p,vgal)/dist/dang2vel;
  pmbrot=product_ecliptic(q,vgal)/dist/dang2vel; 

  /* Take the galactiv rotation and solar motion off the galactic pm */
  pmb = gal[2];
  pml = gal[1];
  pmb = pmb-pmbrot-vsb;
  pml = pml-pmlrot-vsl; 

  *mul=pml;
  *mulerr = pmlerr;
  *mub = pmb;
  *muberr = pmberr;
}

void multiply_ecliptic(double a[3][3],double b[3][3],double c[3][3])
{
  int i,j,k;
  double w,wm[3][3];
  for (i=1;i<=2;i++)
    {
      for (j=1;j<=2;j++)
	{
	  w=0.0;
	  for (k=1;k<=2;k++)
	    w = w +a[i][k]*b[k][j];

	  wm[i][j]=w;
	}
    }
  for (j=1;j<=2;j++)
    {
      for (i=1;i<=2;i++)
	c[i][j]=wm[i][j];
    }
}

void matx2v_ecliptic(double mat[3][3],double *v,double *r)
{
  r[1] = mat[1][1]*v[1]+mat[1][2]*v[2];
  r[2] = mat[2][1]*v[1]+mat[2][2]*v[2];
}

/* Derived from SLA_EQGAL */
void ec2gal_ecliptic(double dr,double dd,double *dl,double *db,double rotmat[3][3])
{
  double v1[4],v2[4],p[4],q[4],pg[4],tp[4],tq[4];
  double rmat[4][4],a,b,c,d,e,f,g,h,i;
  double ce = 0.91748213149438; /* Cos epsilon  */
  double se = 0.39777699580108; /* Sine epsilon */
  double cosb;

  /* Postmultiply equatorial to galactic rotation matrix with */
  /* ecliptic to equatorial rotation matrix */

  a = -0.054875539726;
  b = -0.873437108010;
  c = -0.483834985808;
  d = +0.494109453312;
  e = -0.444829589425;
  f = +0.746982251810;
  g = -0.867666135858;
  h = -0.198076386122;
  i = +0.455983795705;

  rmat[1][1] = a;
  rmat[1][2] = b*ce+c*se;
  rmat[1][3] = -b*se+c*ce;

  rmat[2][1] = d;
  rmat[2][2] = e*ce+f*se;
  rmat[2][3] = -e*se+f*ce;
  
  rmat[3][1] = g;
  rmat[3][2] = h*ce+i*se;
  rmat[3][3] = -h*se+i*ce;

  cosb=cos(dd);
  v1[1] = cos(dr)*cosb;
  v1[2] = sin(dr)*cosb;
  v1[3] = sin(dd);

  /* Calculate orthogonal vectors */

  p[1] = -sin(dr);
  p[2] = cos(dr);
  p[3] = 0;

  q[1] = -sin(dd)*cos(dr);
  q[2] = -sin(dd)*sin(dr);
  q[3] = cos(dd);

  /* Ecliptic to galactic */
  transform_ecliptic(rmat,v1,v2);
  transform_ecliptic(rmat,p,tp);
  transform_ecliptic(rmat,q,tq);

  /* Cartesian to spherical */
  dir2sph_ecliptic(v2,dl,db);

  /* Express in conventional ranges */
  *dl = f_mod_ecliptic(*dl,2.0*M_PI);
  if (*dl < 0) *dl = *dl+M_PI*2.0;

  *db = f_mod_ecliptic(*db,2.0*M_PI);
  if (*db < 0) *db = *db+M_PI*2.0;

  /* Form the p vector in galactic reference frame */
  pg[1] = -sin(*dl);
  pg[2] = cos(*dl);
  pg[3] = 0.0;

  /* Form the proper motion rotation matrix */
  rotmat[1][1] = product_ecliptic(pg,tp);
  rotmat[2][2] = rotmat[1][1];
  rotmat[1][2] = product_ecliptic(pg,tq);
  rotmat[2][1] = -rotmat[1][2];

}
/*  Scalar product of two 3-vectors  (double precision) */
double product_ecliptic(double *va,double *vb)
{
  return va[1]*vb[1]+va[2]*vb[2]+va[3]*vb[3];
}

/* Direction cosines to spherical coordinates (double precision) */
void dir2sph_ecliptic(double *v,double *a, double *b)
{
  double x,y,z,r;

  x=v[1];
  y=v[2];
  z=v[3];
  r=sqrt(x*x+y*y);
  if (r==0)
    *a=0.0;
  else
    *a = atan2(y,x);

  if (z==0)
    *b = 0.0;
  else
    *b = atan2(z,r);
}

/* Performs the 3-D forward unitary transformation: */
void transform_ecliptic(double dm[4][4],double *va,double *vb)
{
  int i,j;
  double w,vw[4];

  for (j=1;j<=3;j++)
    {
      w=0.0;
      for (i=1;i<=3;i++)
	w=w+dm[j][i]*va[i];
      vw[j]=w;
    }

  for (j=1;j<=3;j++)
    vb[j]=vw[j];
}

/* Fortran mod */
double f_mod_ecliptic(double a,double p)
{
  return a - ((int)(a / p) * p);
}


/* Fortran sign */
double f_sign_ecliptic(double a,double b)
{
  if(b>=0.0) 
    return fabs(a);
  else 
    return -fabs(a);
}


