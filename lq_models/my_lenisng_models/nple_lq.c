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
double kappa_nple(double ksi,double b,double s,double alpha) {
	double res;
	res = 0.5*pow(b,2.0-alpha)/pow((s*s+ksi*ksi),1.0-alpha*0.5);
	return res;
}
//----------------------------------------------------------------------------
struct f_params_nple{double x,y,q,b,s,alpha;};
int int_l_nple(double x,double y,double q,double b,double s,double alpha,double ul,double dl,double (*fp)(double, void*),
double * result,double * error){

	struct f_params_nple p = {x,y,q,b,s,alpha};

	gsl_integration_workspace * w
	 = gsl_integration_workspace_alloc (1000);

	gsl_function F;
	F.function = fp;
	F.params = &p;

	gsl_integration_qags (&F, ul, dl, 0, 1e-7, 1000,w, result, error);
	gsl_integration_workspace_free (w);

	return 0;
}
//----------------------------------------------------------------------------
double func_UJ0(double u,double x,double y,double q,double b,double s,double alpha) {
	double kt;
	kt = sqrt(u*(x*x+y*y/(1-(1-q*q)*u)));
	double res;
	res = kappa_nple(kt,b,s,alpha)/sqrt(1.0-(1.0-q*q*u));
	return res;
}

double func_UJ0_gsl(double u,void *params) {

	struct f_params_nple *p = (struct f_params_nple *) params;
	double x = p->x;
	double y = p->y;
	double q = p->q;
	double b = p->b;
	double s = p->s;
	double alpha = p->alpha;

	double kt;
	kt = sqrt(u*(x*x+y*y/(1-(1-q*q)*u)));
	double res;
	res = kappa_nple(kt,b,s,alpha)/sqrt(1.0-(1.0-q*q)*u);
	return res;
}

double UJ0(double x,double y,double q,double b,double s,double alpha) {

	double res = 0.0;
	double abr = 0.0;
	int_l_nple(x,y,q,b,s,alpha,0.000001,1.0,func_UJ0_gsl,&res,&abr);
	return res;
}

double alpha1_nple(double x,double y,double q,double b,double s,double alpha) {
	double res = 0;
	res = q*x*UJ0(x,y,q,b,s,alpha);
	return res;
}

//----------------------------------------------------------------------------
double func_UJ1(double u,double x,double y,double q,double b,double s,double alpha) {
	double kt;
	kt = sqrt(u*(x*x+y*y/(1-(1-q*q)*u)));
	double res;
	res = kappa_nple(kt,b,s,alpha)/pow(1.0-(1.0-q*q*u),1.5);
	return res;
}
double func_UJ1_gsl(double u,void *params) {

	struct f_params_nple *p = (struct f_params_nple *) params;
	double x = p->x;
	double y = p->y;
	double q = p->q;
	double b = p->b;
	double s = p->s;
	double alpha = p->alpha;

	double kt;
	kt = sqrt(u*(x*x+y*y/(1-(1-q*q)*u)));
	double res;
	res = kappa_nple(kt,b,s,alpha)/pow((1.0-(1.0-q*q)*u),1.5);
	return res;
}

double UJ1(double x,double y,double q,double b,double s,double alpha) {

	double res = 0.0;
	double abr = 0.0;
	int_l_nple(x,y,q,b,s,alpha,0.000001,1.0,func_UJ1_gsl,&res,&abr);
	return res;
}

double alpha2_nple(double x,double y,double q,double b,double s,double alpha) {
	double res = 0;
	res = q*y*UJ1(x,y,q,b,s,alpha);
	return res;
}
//----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
	int i,j;
	int Nc = 10;
	double boxsize = 3.0;
	double dsx = boxsize/(double)Nc;
	double x,y,al1,al2;
	double q = 0.8;
	double b = 1.0;
	double s = 0.1;
	double alpha = 1.0;
	for(i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
		x = i*dsx-boxsize*0.5+dsx*0.5;
		y = j*dsx-boxsize*0.5+dsx*0.5;

		al1 = alpha1_nple(x,y,q,b,s,alpha);
		al2 = alpha2_nple(x,y,q,b,s,alpha);
		printf("%lf %lf %lf \n",al1,al2,sqrt(al1*al1+al2*al2));
	}
	return 0;
}
