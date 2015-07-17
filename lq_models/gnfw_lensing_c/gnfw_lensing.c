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
double rho_gnfw(double r,double rhos,double rs,double beta){
	double rx = r/rs;
	double res = 0.0;
	res = rhos/(pow(rx,beta)*pow((1.0+rx),(3.0-beta)));
	return res;
}
//----------------------------------------------------------------------------
double c_m(double m,double z){
	double c0 = 9.0;
	double m_star = 1.5e13;
	double res = 0.0;
	res = c0/(1.0+z)*pow((m/m_star),-0.13);
	return res;
}
//----------------------------------------------------------------------------
double func_gnfw_int(double x,void *params){
	double beta = *(double *) params;
	double res = 0.0;
	res = x*x/(pow(x,beta)*pow((1.0+x),(3.0-beta)));
	return res;
}
//----------------------------------------------------------------------------
double func_gnfw(double x,double beta){
	double res = 0.0;
	double abr = 0.0;
	int_l2p(beta,0,x,func_gnfw_int,&res,&abr);
	return res;
}
//----------------------------------------------------------------------------
//double m200_r200_gnfw(double r,double beta,double c,double z){
//	res = 4.0*M_PI*rhos*rs*rs*rs*func_gnfw(c,beta);
//	return res;
//}
//----------------------------------------------------------------------------
double r200_m200_gnfw(double m,double z){
	double res = 0.0;
	res = pow(((3.0*m*rho_crit(z))/(4.0*M_PI*dv(z))),1.0/3.0);
	return res;
}
//----------------------------------------------------------------------------
double rs_m200(double m,double z){
	double res = 0.0;
	res = 1.0/c_m(m,z)*r200_m200_gnfw(m,z);
	return res;
}
//----------------------------------------------------------------------------
double rhos_m200(double m,double beta,double z){
	double res = 0.0;
	res = dv(z)*rho_crit(z)*c_m(m,z)/(3.0*func_gnfw(c_m(m,z),beta));
	return res;
}
//----------------------------------------------------------------------------
double func_sigma_gnfw(double x,void *params){
	double res = 0.0;
	struct f_params_2 *p
		= (struct f_params_2 *) params;

	double xtmp = p->a;
	double beta = p->b;

	res = pow((x*x+xtmp*xtmp),-beta/2.0)*pow((pow((x*x+xtmp*xtmp),0.5)+1.0),beta-3.0);
	return res;
}
//----------------------------------------------------------------------------
double sigma_gnfw(double x,double m,double beta,double z){
	double res = 0.0;
	double abr = 0.0;
	int_sd3p(x,beta,0.0,func_sigma_gnfw,&res,&abr);
	res = 2.0*rhos_m200(m,beta,z)*rs_m200(m,z)*res;
	return res;
}
//----------------------------------------------------------------------------
double kappa_gnfw(double x,double m,double beta,double z1,double z2){
	double res = 0.0;
	res = sigma_gnfw(x,m,beta,z1)/sigma_crit(z1,z2);
	return res;
}
//----------------------------------------------------------------------------
double func_m2d_gnfw(double x,void *params){
	double res = 0.0;
	struct f_params_3 *p
		=(struct f_params_3 *) params;

	double m = p->a;
	double beta = p->b;
	double z = p->c;

	res = x*sigma_gnfw(x,m,beta,z);
	return res;
}
//----------------------------------------------------------------------------
double m2d_gnfw(double x,double m,double beta,double z){
	double res = 0.0;
	double abr = 0.0;
	int_l4p(m,beta,z,0.0,x,func_m2d_gnfw,&res,&abr);
	res = 2.0*M_PI*rs_m200(m,z)*res;
	return res;
}
//----------------------------------------------------------------------------
double alpha_hat_gnfw(double x,double m,double beta,double z){
	double res = 0.0;
	res = 4.0*M_G*m2d_gnfw(x,m,beta,z)/(vc*vc*rs_m200(m,z)*x);
	return res;
}
//----------------------------------------------------------------------------
double alpha_gnfw(double x,double m,double beta,double z1,double z2){
	double res = 0.0;
	res = alpha_hat_gnfw(x,m,beta,z1)*Da2(z1,z2)/Da(z2);
	return res;
}
//----------------------------------------------------------------------------
double mus_m200(double m,double beta,double z1,double z2){
	double res = 0.0;
	res = 4.0*rhos_m200(m,beta,z1)*rs_m200(m,z1)/sigma_crit(z1,z2);
	return res;
}
//----------------------------------------------------------------------------
double gfunc_int1_gnfw(double x,double beta){
	double res = 0.0;
	double abr = 0.0;
	int_su3p(x,beta,1e-3,func_sigma_gnfw,&res,&abr);
	return res;
}
//----------------------------------------------------------------------------
double gfunc_int2_gnfw(double x,void *params){
	double beta = *(double *) params;
	double res = 0.0;
	res = x*gfunc_int1_gnfw(x,beta);
	return res;
}
//----------------------------------------------------------------------------
double gfunc_gnfw(double x,double beta){
	double res = 0.0;
	double abr = 0.0;
	int_l2p(beta,0.0,x,gfunc_int2_gnfw,&res,&abr);
	return res;
}
//----------------------------------------------------------------------------
double lq_gnfw_gsl(double x,void *params){
	double res = 0.0;
	struct f_params_2 *p
		= (struct f_params_2 *) params;
	double beta = p->a;
	double mus = p->b;
	res = x-mus*gfunc_gnfw(x,beta)/x;
	return res;
}
//----------------------------------------------------------------------------
double lq_gnfw(double x,double beta,double mus){
	double res = 0.0;
	res = x-mus*gfunc_gnfw(x,beta)/x;
	return res;
}
//----------------------------------------------------------------------------
double dy_dx_gnfw_gsl(double x,void *params){
	double res = 0.0;
	double abr = 0.0;
	struct f_params_2 *p
		= (struct f_params_2 *) params;
	double beta = p->a;
	double mus = p->b;
	dif3p(x,beta,mus,lq_gnfw_gsl,&res,&abr);
	return res;
}
//----------------------------------------------------------------------------
double dy_dx_gnfw(double x,double beta,double mus){
	double res = 0.0;
	double abr = 0.0;
	dif3p(x,beta,mus,lq_gnfw_gsl,&res,&abr);
	return res;
}
//----------------------------------------------------------------------------
double x_zero(double beta,double mus){
	double res = 0.0;
	f_root3p(beta,mus,0.00001,5.0,lq_gnfw_gsl,&res);
	return res;
}
//----------------------------------------------------------------------------
double x_cr(double beta,double mus){
	double res = 0.0;
	f_root3p(beta,mus,0.00001,x_zero(beta,mus),dy_dx_gnfw_gsl,&res);
	return res;
}
//----------------------------------------------------------------------------
double y_cr(double beta,double mus){
	double res = 0.0;
	res = lq_gnfw(x_cr(beta,mus),beta,mus);
	return res;
}
//----------------------------------------------------------------------------
double cs_gnfw(double m,double mus,double beta,double z){
	double res = 0.0;
	res = M_PI*pow((y_cr(beta,mus)*rs_m200(m,z)),2.0);
	//res = M_PI*pow(y_cr(beta,mus),2.0);
	return res;
}
//----------------------------------------------------------------------------
void main(int argc, char *argv[]){
	double xt = (double)atof(argv[1]);
	//int i;
	//int n = 100;
	//double ap = 1.1;
	//double mp = 1.0;
	double yt;
	//for(i=0;i<n;i++){
	//	xt = (double)(i+0.5)*1.0/(double)n-0.5;
	//	yt = lq_gnfw(xt,1.5,1.0);
	//	//printf("%lf %lf \n",xt,yt);
	//	printf("%lf ",yt);
	//}
	//xt = x_cr(ap,mp);
	//yt = cs_gnfw(1e11,1.0,1.3,0.2);
	//yt = y_cr(ap,mp);
	yt = lq_gnfw(xt,1.5,1.0);
	printf("%lf %lf\n",xt,yt);
}
