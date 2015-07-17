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
//----------------------------------------------------------------------------
double kappa_nie(double ksi,double b,double s) {
	double res;
	res = 0.5*b/sqrt(s*s+ksi*ksi);
	return res;
}
//----------------------------------------------------------------------------
double alpha1_nie(double x,double y,double q,double b,double s) {
	double phi = sqrt(q*q*(s*s+x*x)+y*y);
	double res = 0;
	res = b*q/sqrt(1-q*q)*atan(sqrt(1.0-q*q)*x/(phi+s));
	return res;
}

double alpha2_nie(double x,double y,double q,double b,double s) {
	double phi = sqrt(q*q*(s*s+x*x)+y*y);
	double res = 0;
	res = b*q/sqrt(1.0-q*q)*atanh(sqrt(1.0-q*q)*y/(phi+q*q*s));
	return res;
}
//----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
	int i,j;
	int Nc = 10;
	double boxsize = 3.0;
	double dsx = boxsize/(double)Nc;
	double x,y,al1,al2;
	double q = 0.9999999;
	double b = 1.0;
	double s = 0.0000001;
	double alpha = 1.0;
	for(i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
		x = i*dsx-boxsize*0.5+dsx*0.5;
		y = j*dsx-boxsize*0.5+dsx*0.5;

		al1 = alpha1_nie(x,y,q,b,s);
		al2 = alpha2_nie(x,y,q,b,s);
		printf("%lf %lf %lf \n",al1,al2,sqrt(al1*al1+al2*al2));
	}
	return 0;
}
