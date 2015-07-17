#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//-------------------------------------------------------------------------------
double func_deriv(double x){
	double res;
	res = x*x*x;
	return res;
}
double func_deriv_d(double x){
	double res;
	res = 3.0*x*x;
	return res;
}
double ld_k(double x,double h,int n,double (*fp)(double)){
	int k;
	double res= 0.0;
	double res_num = 0.0;//numerator
	double res_den = 0.0;//denominator
	for(k=-n;k<=n;k++) res_num += k*fp(x+(double)k*h);
	for(k=0;k<n;k++) res_den += 2.0*pow(((double)k+1.0),2.0)*h;
	res = res_num/res_den;
	return res;
}
//-------------------------------------------------------------------------------
void main(){
	double r = 0.0;
	double a = 1.451;
	double stp = 1e-5;
	int ord = 1; //np = 2*n+1
	r = ld_k(a,stp,ord,func_deriv);
	printf("%8.13lf \n", r);
	printf("%8.13lf \n", fabs(func_deriv_d(a)-r));
}
