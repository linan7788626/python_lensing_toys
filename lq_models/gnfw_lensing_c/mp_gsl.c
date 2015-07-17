#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_integration.h>

//----------------------------------------------------------------------------
struct my_f_params {int a; int b;}; 

void matrixMaker(int n, mat & H, mat & Q, vec & d){
        
        gsl_function F1; 
        struct my_f_params alpha = {2,2};               
        F1.function = &f1; 
        F1.params = &alpha;
        double result1,result2,error;
        gsl_integration_workspace * w = gsl_integration_workspace_alloc 
(100000); 
        int size = n; //Size
        
        for(int n = 0; n < size; n++)
        for(int m = 0; m < size; m++)   {
                if(abs(m-n) < 7){
                        alpha.a = n;
                        alpha.b = m;
                        gsl_integration_qagiu (&F1, 0, 0, 1e-9, 10000, w, 
&result1, &error); 
                        gsl_integration_qagil (&F1, 0, 0, 1e-9, 10000, w, 
&result2, &error);
                        H(n,m) = result1 + result2;
                }
                else
                        H(n,m) = 0;
        }
        eig_sym(H,d,Q);
        gsl_integration_workspace_free (w);
}

and the function I integrate is

double f1 (double x, void * p) { //Deriviative part of the integral
        struct my_f_params * params = (struct my_f_params *)p;
        int n = (params->a);
        int m = (params->b);
        
        gsl_function w;
        double result, abserr;
        w.function = &w1;
        w.params = 0;
        gsl_deriv_central(&w,x,1e-8,&result,&abserr);
        
        double f = exp(-(x*x))*hermitecalculate(m,x)*( (-1.0)*((4*n*n - 
4*n)*hermitecalculate(n-2,x) - 4*n*hermitecalculate(n-1,x)*x - 
hermitecalculate(n,x) + hermitecalculate(n,x)*x*x)); 
        f += exp(-x*x)*hermitecalculate(n,x)*hermitecalculate(m,x)*( w1(x,0) * 
w1(x,0) + result);
        return 
f/(sqrt(gsl_sf_fact(n)*sqrt(pi)*pow(2.0,n))*sqrt(gsl_sf_fact(m)*sqrt(pi)*pow(2.0,m)));
 
}
