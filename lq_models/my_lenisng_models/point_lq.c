#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>

#include "gsl_myfuncs.h"
//----------------------------------------------------------------------------
double sigma_crit(double z1,double z2){
	double res = 0.0;
	res = vc*vc*Da(z2)/(4.0*M_PI*M_G*Da(z1)*Da2(z1,z2));
	return res;
}

double re_m(double m,double z1,double z2) {
	double res = 0;
	res = sqrt(M_G*m/(vc*vc)*Da2(z1,z2)/(Da(z1)*Da(z2)));
	return res;
}
//----------------------------------------------------------------------------
double alpha1_point(double x1,double x2,double re) {
	double xq = sqrt(x1*x1+x2*x2);
	printf("%lf %lf %lf --------------\n",x1,x2,xq);
	double req = re*re;
	double res = 0;
	res = x1*req/xq;
	return res;
}
//----------------------------------------------------------------------------
double alpha2_point(double x1,double x2,double re) {
	double xq = sqrt(x1*x1+x2*x2);
	printf("%lf %lf %lf ##############\n",x1,x2,xq);
	double req = re*re;
	double res = 0;
	res = x2*req/xq;
	return res;
}
//----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
	int i,j;
	int Nc = 10;
	double boxsize = 3.0;
	double dsx = boxsize/(double)Nc;
	double x,y,al1,al2;
	double re = 1.0;
	for(i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
		x = (double)i*dsx-boxsize*0.5+dsx*0.5;
		y = (double)j*dsx-boxsize*0.5+dsx*0.5;

		al1 = alpha1_point(x,y,re);
		al2 = alpha2_point(x,y,re);
		printf("%lf %lf %lf \n",x,y,sqrt(al1*al1+al2*al2));
	}
	return 0;
}
