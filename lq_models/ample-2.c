/*============================================================================
+
 NAME:
   ample.c
   
 CONTAINS: 
   ample_ei_minus_ln_minus_gamma
   ample_psi_sersic
   ample_psi_sersic_general 
   ample_psi_NFW
   ample_psi_NFW_trunc
   ample_psi_NFW_trunc2
   ample_psi_NFW_trunc_hard
   ample_rho_NFW
   ample_rho_NFW_trunc
   ample_rho_NFW_trunc2  
   ample_rho_NFW_trunc_hard
   
 PURPOSE:
   Functions to compute the potential psi, and its first three derivatives
   with respect to u = r^2, for the  Analytic Models of Plausible LEnses of 
   Baltz, Marshall and Oguri 2007.

 COMMENTS:
   Compile with e.g.
     gcc -o ample ample.c -lm
     
 OUTPUTS:
   stdout       Test program main tabulates some potentials and derivatives.

 BUGS:
   - No elliptical quadratic form calculator for u - make your own!

 REVISION HISTORY:
   2007-04-30  Copyright Edward A. Baltz (KIPAC)

-
============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define EULER_GAMMA 0.5772156649015328606

/*--------------------------------------------------------------------------*/

double ample_ei_minus_ln_minus_gamma ( double y )
{
  // we want Ei(y)-ln(-y)-gamma =
  //        -E1(-y)-ln(-y)-gamma
  double x = - y;
  double term, sum;
  int i;
  
  if ( x <= 32.0 ) {
    term = - x;
    sum = term;
    for ( i = 2 ; i < 120 ; ++i ) {
      term *= - x * ( i - 1.0 ) / ( i * i );
      sum += term;
      if ( fabs ( term / sum ) < 1.e-12 ) break;
    }
    return sum;
  }
  else {
    term = 1;
    sum = term;
    for ( i = 1 ; i < 120 ; ++i ) {
      term *= - i / x;
      sum += term;
      if ( fabs ( term / sum ) < 1.e-12 ) break;
    }
    return - sum * exp ( - x ) / x - log ( x ) - EULER_GAMMA;
  }
}

/*--------------------------------------------------------------------------*/

/* psi is an array of 4 doubles to store the potential and its
   first 3 derivatives

   u = r^2 in units of the scale radius

   two_n is twice the Sersic index, as an integer

   This function has hard-coded constants for n <= 10.  For larger n,
   the function ample_psi_sersic_general is called */
void ample_psi_sersic ( double psi[], double u, int two_n )
{
  void ample_psi_sersic_general ( double *, double, int );
  if ( two_n > 20 ) {
    ample_psi_sersic_general ( psi, u, two_n );
    return;
  }
  double lnu = log ( u );
  double u_2nth = exp ( lnu / two_n );
  double exp_term = exp ( - u_2nth );
  /* these are the values of u = r^2 the enclose half of the projected mass
     for Sersic indices from 1/2 to 10 in increments of 1/2 */
  /*  const double radii[] =
      { 6.93147179e-01, 2.81684865e+00,
      1.91211315e+01, 1.81819065e+02,
      2.22334562e+03, 3.32332207e+04,
      5.87104153e+05, 1.19680365e+07,
      2.76502489e+08, 7.13982261e+09,
      2.03773065e+11, 6.36970374e+12,
      2.16424702e+14, 7.94182953e+15,
      3.13017949e+17, 1.31882041e+19,
      5.91499659e+20, 2.81370464e+22,
      1.41493713e+24, 7.49996651e+25 }; */
  const double fact_inv[21] =
    { 1.000000000000000000e+00, 1.000000000000000000e+00,
      5.000000000000000000e-01, 1.666666666666666574e-01,
      4.166666666666666435e-02, 8.333333333333333218e-03,
      1.388888888888888942e-03, 1.984126984126984125e-04,
      2.480158730158730157e-05, 2.755731922398589251e-06,
      2.755731922398589357e-07, 2.505210838544172022e-08,
      2.087675698786810019e-09, 1.605904383682161593e-10,
      1.147074559772972612e-11, 7.647163731819817415e-13,
      4.779477332387385885e-14, 2.811457254345520993e-15,
      1.561920696858622774e-16, 8.220635246624331036e-18,
      4.110317623312165326e-19 };
  const double aa[][20] = 
    { { 0.000000000000000000e+00 },
      { 1.000000000000000000e+00 },
      { 1.500000000000000000e+00, 5.000000000000000000e-01 },
      { 1.833333333333333259e+00, 8.333333333333332593e-01,
	1.666666666666666574e-01 },
      { 2.083333333333333037e+00, 1.083333333333333259e+00,
	2.916666666666666297e-01, 4.166666666666666435e-02 },
      { 2.283333333333333215e+00, 1.283333333333333215e+00,
	3.916666666666666075e-01, 7.499999999999999722e-02,
	8.333333333333333218e-03 },
      { 2.449999999999999734e+00, 1.449999999999999956e+00,
	4.749999999999999223e-01, 1.027777777777777735e-01,
	1.527777777777777901e-02, 1.388888888888888725e-03 },
      { 2.592857142857142527e+00, 1.592857142857142749e+00,
	5.464285714285713746e-01, 1.265873015873015817e-01,
	2.123015873015872759e-02, 2.579365079365079309e-03,
	1.984126984126984125e-04 },
      { 2.717857142857142527e+00, 1.717857142857142749e+00,
	6.089285714285713746e-01, 1.474206349206348965e-01,
	2.643849206349205977e-02, 3.621031746031746178e-03,
	3.720238095238095032e-04, 2.480158730158730157e-05 },
      { 2.828968253968253688e+00, 1.828968253968253910e+00,
	6.644841269841269549e-01, 1.659391534391534417e-01,
	3.106812169312169261e-02, 4.546957671957672573e-03,
	5.263447971781304606e-04, 4.684744268077601557e-05,
	2.755731922398588828e-06 },
      { 2.928968253968253777e+00, 1.928968253968253999e+00,
	7.144841269841269993e-01, 1.826058201058201047e-01,
	3.523478835978835488e-02, 5.380291005291005201e-03,
	6.652336860670193764e-04, 6.668871252204586360e-05,
	5.235890652557319408e-06, 2.755731922398589357e-07 },
      { 3.019877344877344605e+00, 2.019877344877345049e+00,
	7.599386724386724135e-01, 1.977573352573352428e-01,
	3.902266714766714634e-02, 6.137866762866763318e-03,
	7.914963123296456932e-04, 8.472623055956390111e-05,
	7.490580407247074095e-06, 5.260942760942761644e-07,
	2.505210838544172353e-08 },
      { 3.103210678210678086e+00, 2.103210678210678530e+00,
	8.016053391053390431e-01, 2.116462241462241378e-01,
	4.249488936988936316e-02, 6.832311207311207897e-03,
	9.072370530703864925e-04, 1.012606220939554400e-04,
	9.557379349046014763e-06, 7.557386029608252334e-07,
	4.801654107209663374e-08, 2.087675698786810019e-09 },
      { 3.180133755133755180e+00, 2.180133755133755624e+00,
	8.400668775668774790e-01, 2.244667369667369405e-01,
	4.570001757501757078e-02, 7.473336848336849247e-03,
	1.014074659907993493e-03, 1.165231373564707161e-04,
	1.146519375686042258e-05, 9.677179816068704871e-07,
	6.921447893670116573e-08, 4.014760959205403309e-09,
	1.605904383682161593e-10 },
      { 3.251562326562326799e+00, 2.251562326562327243e+00,
	8.757811632811631775e-01, 2.363714988714988308e-01,
	4.867620805120804334e-02, 8.068574943574944106e-03,
	1.113281009114342563e-03, 1.306954729573777513e-04,
	1.323673570697380199e-05, 1.164555976063912455e-06,
	8.889827838240536786e-08, 5.804197272451240468e-09,
	3.097101311387025892e-10, 1.147074559772972451e-11 },
      { 3.318228993228993673e+00, 2.318228993228994117e+00,
	9.091144966144965034e-01, 2.474826099826099579e-01,
	5.145398582898581819e-02, 8.624130499130499769e-03,
	1.205873601706935246e-03, 1.439229861848909608e-04,
	1.489017486041295486e-05, 1.348271437557151711e-06,
	1.072698245317292907e-07, 7.474337831480688979e-09,
	4.488885110578232984e-10, 2.217677482227747050e-11,
	7.647163731819817415e-13 },
      { 3.380728993228993673e+00, 2.380728993228994117e+00,
	9.403644966144965034e-01, 2.578992766492765876e-01,
	5.405815249565248948e-02, 9.144963832463832987e-03,
	1.292679157262490710e-03, 1.563237798356846217e-04,
	1.644027406676216248e-05, 1.520504682707063645e-06,
	1.244931490467204709e-07, 9.040094605570795666e-09,
	5.793682422319988729e-10, 3.221367722029098046e-11,
	1.481637973040089422e-12, 4.779477332387385885e-14 },
      { 3.439552522640758170e+00, 2.439552522640758614e+00,
	9.697762613203788629e-01, 2.677031982179040592e-01,
	5.650913288780935045e-02, 9.635159910895205529e-03,
	1.374378503667719395e-03, 1.679951150364315999e-04,
	1.789919096685553136e-05, 1.682606560495215978e-06,
	1.407033368255357149e-07, 1.051374804000854557e-08,
	7.021726951018113303e-10, 4.166017359489193176e-11,
	2.156387714083014631e-12, 9.277808939340220104e-14,
	2.811457254345520993e-15 },
      { 3.495108078196313528e+00, 2.495108078196313972e+00,
	9.975540390981566530e-01, 2.769624574771633041e-01,
	5.882394770262416861e-02, 1.009812287385816916e-02,
	1.451538997494879928e-03, 1.790180427260259501e-04,
	1.927705692805482853e-05, 1.835702778406248832e-06,
	1.560129586166389950e-07, 1.190553183919975059e-08,
	8.181546783677452547e-10, 5.058186461534838003e-11,
	2.793651358401332883e-12, 1.352623323479567310e-13,
	5.466722439005179511e-15, 1.561920696858622774e-16 },
      { 3.547739657143682113e+00, 2.547739657143682557e+00,
	1.023869828571840834e+00, 2.857343873017247349e-01,
	6.101693015876451243e-02, 1.053671936508623896e-02,
	1.524638412699558373e-03, 1.894608163266942762e-04,
	2.058240362813837268e-05, 1.980741300637753808e-06,
	1.705168108397894555e-07, 1.322406385948615805e-08,
	9.280323467249457385e-10, 5.903399295051765497e-11,
	3.397374810913423201e-12, 1.755105625154294357e-13,
	7.982236824472223550e-15, 3.041635041251002114e-16,
	8.220635246624329496e-18 } };
  
  double sum, suma;
  double exp_term_sum;

  int j;
  if ( two_n == 2 || two_n == 1 )
    suma = aa[two_n-1][0];
  else
    suma = u_2nth * aa[two_n-1][two_n-2] + aa[two_n-1][two_n-3];
  
  for ( j = two_n - 4 ; j >= 0 ; --j )
    suma = u_2nth * suma + aa[two_n-1][j];

  if ( two_n == 1 )
    sum = 1.0;
  else
    sum = u_2nth * fact_inv[two_n-1] + fact_inv[two_n-2];
  
  for ( j = two_n - 3 ; j >= 0 ; --j )
    sum = u_2nth * sum + fact_inv[j];
  exp_term_sum = exp_term * sum;

  psi[0] = ( - ample_ei_minus_ln_minus_gamma ( - u_2nth )
	     - aa[two_n-1][0] + exp_term * suma ) * two_n;

  double u_inv = 1.0 / u;

  psi[1] = ( 1.0 - exp_term_sum ) * u_inv;
  
  psi[2] = ( - ( 1.0 - exp_term_sum ) * u_inv
	     + exp_term * fact_inv[two_n] ) * u_inv;
  
  psi[3] = ( 2.0 * ( 1.0 - exp_term_sum ) * u_inv
	     - exp_term * fact_inv[two_n] *
	     ( 2.0 + u_2nth / two_n ) ) * u_inv * u_inv;

  return;
}

/*--------------------------------------------------------------------------*/

void ample_psi_sersic_general ( double psi[], double u, int two_n )
{
  double lnu = log ( u );
  double u_2nth = exp ( lnu / two_n );
  double exp_term = exp ( - u_2nth );
  double a[two_n-1];
  double fact[two_n+1];
  double sum, suma;
  double exp_term_sum;
  
  int j, k;
  for ( j = two_n - 2 ; j >= 0 ; --j ) {
    a[j] = 0.0;
    for ( k = j + 1 ; k < two_n ; ++k )
      a[j] += 1.0 / k;
  }
  fact[0] = 1.0;
  for ( j = 1 ; j < two_n + 1 ; ++j )
    fact[j] = j * fact[j-1];
  
  if ( two_n == 1 )
    suma = 0.0;
  else if ( two_n == 2 )
    suma = a[0];
  else
    suma = u_2nth * a[two_n-2] / fact[two_n-2] + a[two_n-3] / fact[two_n-3];
  
  for ( j = two_n - 4 ; j >= 0 ; --j )
    suma = u_2nth * suma + a[j] / fact[j];

  if ( two_n == 1 )
    sum = 1.0;
  else
    sum = u_2nth / fact[two_n-1] + 1.0 / fact[two_n-2];
  
  for ( j = two_n - 3 ; j >= 0 ; --j )
    sum = u_2nth * sum + 1.0 / fact[j];
  exp_term_sum = exp_term * sum;

  psi[0] = ( - ample_ei_minus_ln_minus_gamma ( - u_2nth )
	     - a[0] + exp_term * suma ) * two_n;

  psi[1] = ( 1.0 - exp_term_sum ) / u;

  psi[2] = ( - ( 1.0 - exp_term_sum ) / u + exp_term / fact[two_n] ) / u;

  psi[3] = ( 2.0 * ( 1.0 - exp_term_sum ) / u - exp_term / fact[two_n] *
	     ( 2.0 + u_2nth / two_n ) ) / ( u * u );

  return;
}

/*--------------------------------------------------------------------------*/

/* psi is an array of 4 doubles to store the potential and its
   first 3 derivatives

   u = r^2 in units of the scale radius

   C is the truncation radius in units of the scale radius */

void ample_psi_NFW ( double psi[], double u )
{
  // the profile assumed is rho0/(4pi)/(x(1+x)^2)
  
  if ( fabs ( u - 1.0 ) < 0.00008 ) {
    double psi1[4], psi2[4], frac;
    int i;
    ample_psi_NFW ( psi1, 1.0001 );
    ample_psi_NFW ( psi2, 0.9999 );
    frac = ( u - 0.9999 ) / 0.0002;
    for ( i = 0 ; i < 7 ; ++i )
      psi[i] = psi1[i] + frac * ( psi2[i] - psi1[i] );
    return;
  }

  double um1 = u - 1.0;
  double u_um1 = u * um1;
  double u_um1_inv = 1.0 / u_um1;
  double u_inv = um1 * u_um1_inv;
  double rootu = sqrt ( u );
  double lnrootu = log ( 0.5 * rootu );

  complex double arccosu = cacos ( 1.0 / rootu );
  complex double sqrtum1, arccossqrt;

  psi[0] = lnrootu * lnrootu + arccosu * arccosu;
  
  sqrtum1 = csqrt ( u - 1.0 );
  arccossqrt = arccosu / sqrtum1;
  if ( u < 1.0 ) arccossqrt = - arccossqrt;
  
  psi[1] = ( lnrootu + arccossqrt ) * u_inv;
  
  double um1_ln = um1 * lnrootu;
  psi[2] = 0.5 * ( ( 2.0 - 3.0 * u ) * arccossqrt +
		   u - 2.0 * um1_ln ) * u_inv * u_um1_inv;
  
  double u_um1_2_inv = u_um1_inv * u_um1_inv;
  double um1_2_ln = um1_ln * um1;
  psi[3] = 0.25 *
    ( ( 3.0 - 6.0 * u ) * u +
      ( 8.0 + u * ( 15.0 * u - 20.0 ) ) * arccossqrt +
      8.0 * um1_2_ln ) * u_inv * u_um1_2_inv;
  
  return;
}

/*--------------------------------------------------------------------------*/

void ample_psi_NFW_trunc ( double psi[], double u, double C )
{
  // the profile assumed is rho0/(4pi)/(x(1+x)^2)*C^2/(C^2+x^2)
  
  if ( fabs ( u - 1.0 ) < 0.00008 ) {
    double psi1[4], psi2[4], frac;
    int i;
    ample_psi_NFW_trunc ( psi1, 1.0001, C );
    ample_psi_NFW_trunc ( psi2, 0.9999, C );
    frac = ( u - 0.9999 ) / 0.0002;
    for ( i = 0 ; i < 7 ; ++i )
      psi[i] = psi1[i] + frac * ( psi2[i] - psi1[i] );
    return;
  }
  
  double lnC = log ( C );
  double um1 = u - 1.0;
  double rootu = sqrt ( u );
  double lnu = log ( u );
  double u_inv = 1.0 / u;
  double u2 = u * u;
  double u2_inv = u_inv * u_inv;
  
  complex double arccosu = cacos ( 1.0 / rootu );
  complex double sqrtum1, arccossqrt;
  
  sqrtum1 = csqrt ( um1 );
  if ( u < 1.0 ) sqrtum1 = - sqrtum1;
  arccossqrt = arccosu / sqrtum1;
  
  double C2 = C * C;
  double C2m1 = C2 - 1.0;
  double C2p1 = C2 + 1.0;
  double C2p1_2 = C2p1 * C2p1;
  double C2p1_2_inv = 1.0 / C2p1_2;
  double rootC2u = sqrt ( C2 + u );
  double rootC2umC = rootC2u - C;
  double rootC2umC_2 = rootC2umC * rootC2umC;
  double biglog = log ( ( rootC2u - C ) / rootu );
  double C2m1_log = C2m1 * biglog;

  psi[0] = ( 2.0 * C2 * M_PI * ( C - rootC2u +
				 C * log ( rootC2u + C ) ) +
	     ( 2.0 * C * rootC2u + C2 * biglog ) * C2m1_log +
	     4.0 * C2 * arccossqrt * um1 +
	     C2 * C2m1 * arccosu * arccosu +
	     C2 * ( C2m1 * lnC - C2 - 1.0 ) * lnu -
	     C2 * ( C2m1 * lnC * ( lnC + 2.0 * M_LN2 ) +
		    2.0 * ( lnC - M_LN2 ) -
		    2.0 * C * ( C - M_PI ) * ( lnC + M_LN2 ) ) ) * C2p1_2_inv;
  
  psi[1] = C2 * ( ( C2 + 2.0 * u - 1.0 ) * arccossqrt + C * M_PI +
		  C2m1 * lnC +
		  rootC2u / C * ( - C * M_PI + C2m1_log ) ) *
    C2p1_2_inv * u_inv;
  
  psi[2] = 0.5 * C2 * C2p1_2_inv * u2_inv *
    ( 2.0 * ( 1.0 - u - C2 ) * arccossqrt
      - 2.0 * C2m1 * lnC
      + C2p1 * u / um1 * ( 1.0 - arccossqrt )
      - C2m1_log * ( 2.0 * C2 + u ) / ( C * rootC2u )
      + M_PI / rootC2u * rootC2umC_2 );

  psi[3] = 0.5 *
    ( 3.0 * C2p1_2 * u /
      ( 2.0 * ( C2 + u ) * um1 * um1 ) * ( u * arccossqrt - 1.0 )
      + u / ( 2.0 * ( C2 + u ) * um1 ) * 
      ( ( 3.0 + C2 - 2.0 * u ) * u * arccossqrt 
	- C2p1 * ( u + 2.0 * C2p1 ) )
      + 2.0 * C2p1 / um1 * ( arccossqrt - u )
      + ( 4.0 * u + 6.0 * C2 - 2.0 ) * arccossqrt
      + C2m1 / ( 2.0 * C * rootC2u * ( C2 + u ) ) *
      ( 3.0 * u2 + 12.0 * C2 * u + 8.0 * C2 * C2 ) * biglog
      + 4.0 * C2m1 * lnC
      - 2.0 * M_PI * rootC2umC_2 / rootC2u
      + M_PI * u2 / ( 2.0 * rootC2u * ( C2 + u ) ) ) *
    C2p1_2_inv * u2_inv * u_inv * C2;

  return;
}

/*--------------------------------------------------------------------------*/

void ample_psi_NFW_trunc2 ( double psi[], double u, double C )
{
  // the profile assumed is rho0/(4pi)/(x(1+x)^2)*C^4/(C^2+x^2)^2
  
  if ( fabs ( u - 1.0 ) < 0.00008 ) {
    double psi1[4], psi2[4], frac;
    int i;
    ample_psi_NFW_trunc2 ( psi1, 1.0001, C );
    ample_psi_NFW_trunc2 ( psi2, 0.9999, C );
    frac = ( u - 0.9999 ) / 0.0002;
    for ( i = 0 ; i < 7 ; ++i )
      psi[i] = psi1[i] + frac * ( psi2[i] - psi1[i] );
    return;
  }
  
  double lnC = log ( C );
  double ln2C = lnC + M_LN2;
  double um1 = u - 1.0;
  double rootu = sqrt ( u );
  double lnu = log ( u );
  double u2 = u * u;
  double u3 = u2 * u;

  complex double arccosu = cacos ( 1.0 / rootu );
  complex double sqrtum1, arccossqrt;
  
  sqrtum1 = csqrt ( um1 );
  if ( u < 1.0 ) sqrtum1 = - sqrtum1;
  arccossqrt = arccosu / sqrtum1;
  
  double C2 = C * C;
  double C4 = C2 * C2;
  double C6 = C4 * C2;
  double C8 = C4 * C4;
  double rootC2u = sqrt ( C2 + u );
  double messylog = log ( ( rootC2u - C ) / rootu );
  
  psi[0] = ( 2.0 * C2 * C * M_PI *
	     ( - 4.0 * C * rootC2u +
	       ( 3.0 * C2 - 1.0 ) * log ( rootC2u + C ) ) +
	     C * ( 6.0 * C4 - 12.0 * C2 - 2.0 ) *
	     rootC2u * messylog +
	     2.0 * C4 * ( C2 - 3.0 ) * messylog * messylog +
	     16.0 * C4 * arccossqrt * um1 +
	     2.0 * C4 * ( C2 - 3.0 ) * arccosu * arccosu +
	     C2 * ( 2.0 * C2 * ( C2 - 3.0 ) * lnC - 3.0 * C4
		    - 2.0 * C2 + 1.0 ) * lnu +
	     2.0 * C2 * ( C2 * ( 4.0 * C * M_PI + ( C2 - 3.0 ) * M_LN2 * M_LN2
				 + 8.0 * M_LN2 )
			  - ln2C * ( 1.0 + 6.0 * C2 - 3.0 * C4
				     + C2 * ( C2 - 3.0 ) * ln2C )
			  - C* M_PI * ( 3.0 * C2 - 1.0 ) * ln2C ) ) /
    ( 2.0 * ( C2 + 1.0 ) * ( C2 + 1.0 ) * ( C2 + 1.0 ) );

  psi[1] = C4 / ( 2.0 * ( C2 + 1.0 ) * ( C2 + 1.0 ) * ( C2 + 1.0 ) * u ) *
    ( 2.0 * ( C2 + 4.0 * u - 3.0 ) * arccossqrt 
      + ( M_PI * ( 3.0 * C2 - 1.0 ) + 2.0 * C * ( C2 - 3.0 ) * lnC ) / C
      + ( - C2 * C * M_PI * ( 4.0 * u + 3.0 * C2 - 1.0 )
	  + ( 2.0 * C6 - 6.0 * C4
	      + u * ( 3.0 * C4 - 6.0 * C2 - 1.0 ) ) * messylog ) /
      ( C2 * C * rootC2u ) );

  psi[2] = C4 / ( 4.0 * ( C2 + 1.0 ) * ( C2 + 1.0 ) * ( C2 + 1.0 ) *
		  u2  ) *
    ( ( 10.0 - 8.0 * u - 6.0 * C2 ) * arccossqrt - 4.0 * ( C2 - 3.0 ) * lnC
      + ( 2.0 * C6 + 3.0 * C4 * ( u - 2 ) - u - 6.0 * C2 * u ) /
      ( C2 * ( C2 + u ) )
      + ( 2.0 * u * ( C2 + u ) * ( 3.0 * C4 - 6.0 * C2 - 1.0 )
	  - ( 3.0 * u + 2.0 * C2 ) *
	  ( 2.0 * C6 + 3.0 * C4 * ( u - 2 ) - u
	    - 6.0 * C2 * u ) ) / ( C2 * C * ( C2 + u ) * rootC2u ) * messylog
      - 2.0 * M_PI / C * ( 3.0 * C2 - 1.0 )
      + M_PI / ( ( C2 + u ) * rootC2u ) *
      ( 2.0 * ( C2 + u ) * ( 3.0 * C2 - 1.0 )
	+ u * ( 4.0 * u + 3.0 * C2 - 1.0 ) )
      + 2.0 * ( C2 + 1.0 ) / um1 * ( 1.0 - arccossqrt ) + 8.0 );

  psi[3] = 1.0 / ( 8.0 * ( C2 + 1.0 ) * ( C2 + 1.0 ) * ( C2 + 1.0 ) *
		   u3 ) *
    ( - 8.0 * C4 * ( 4.0 * u + C2 - 3.0 ) / um1
      + 4.0 * C2 * ( -2.0 * C6 - 3.0 * C4 * ( u - 2.0 )
		     + u + 6.0 * C2 * u ) / ( C2 + u )
      - C4 * u / ( um1 * um1 * ( C2 + u ) * ( C2 + u ) ) *
      ( C6 * ( 2.0 + 4.0 * u )
	+ C4 * ( 6.0 + 8.0 * u + 4.0 * u2 )
	+ 2.0 * ( u3 + u + 1.0 )
	+ C2 * ( 6.0 + 7.0 * u + 2.0 * u2 + 3.0 * u3 ) )
      + 2.0 * C4 * arccossqrt / ( um1 * um1 ) *
      ( C2 * ( 8.0 - 20.0 * u + 15.0 * u2 )
	+ 3.0 * ( - 8.0 + 20.0 * u - 15.0 * u2 + 4.0 * u3 ) )
      + C2 / ( rootC2u * ( C2 + u ) * ( C2 + u ) ) *
      ( -24.0 * C8 * M_PI + C6 * M_PI * ( 8.0 - 60.0 * u )
	+ 24.0 * C6 * C * M_PI * rootC2u + u2 * rootC2u
	- 8.0 * C * M_PI * rootC2u * u * u
	+ 8.0 * C2 * C * M_PI * u * rootC2u * ( 3.0 * u - 2.0 )
	- 3.0 * C2 * M_PI * u2 * ( 4.0 * u - 5.0 )
	+ 8.0 * C4 * C * M_PI * rootC2u * ( 6.0 * u - 1.0 )
	- 5.0 * C4 * u * M_PI * ( 9.0 * u - 4.0 )
	+ 16.0 * C2 * ( C2 - 3.0 ) * rootC2u * ( C2 + u ) * ( C2 + u ) * lnC )
      + messylog / ( rootC2u * ( C2 + u ) * ( C2 + u ) ) *
      ( 16.0 * C8 * C2 * C + 30.0 * C6 * C * ( u - 4.0 ) * u 
	+ 9.0 * C4 * C * ( u - 10.0 ) * u2 - 3.0 * C * u3
	- 18.0 * C2 * C * u3 + 8.0 * C8 * C * ( 5 * u - 6.0 ) ) );
  
  return;
}

/*--------------------------------------------------------------------------*/

void ample_psi_NFW_trunc_hard ( double psi[], double u, double C )
{
  // the profile assumed is rho0/(4pi)/(x(1+x)^2)
  
  if ( fabs ( u - 1.0 ) < 0.00008 ) {
    double psi1[4], psi2[4], frac;
    int i;
    ample_psi_NFW_trunc_hard ( psi1, 1.0001, C );
    ample_psi_NFW_trunc_hard ( psi2, 0.9999, C );
    frac = ( u - 0.9999 ) / 0.0002;
    for ( i = 0 ; i < 7 ; ++i )
      psi[i] = psi1[i] + frac * ( psi2[i] - psi1[i] );
    return;
  }
  
  double mass = log ( 1.0 + C) - C / ( 1.0 + C );
  double C2 = C * C;
  double rootC2u, arctanhroot, um1;
  complex double rootu, arctanfunc;

  if ( u < C2 ) {
    rootC2u = sqrt ( C2 - u );
    rootu = csqrt ( u - 1.0 );
    arctanhroot = atanh ( rootC2u / C );
    um1 = u - 1.0;
    arctanfunc = 1.0 / rootu * ( catan ( rootC2u / rootu ) -
				 catan ( rootC2u / ( C * rootu ) ) );
    //    printf ( "\n%.4e\n\n", cimag ( arctanfunc ) );
    psi[0] = 0.0; // requires at least polylogs
    psi[1] = ( mass + rootC2u / ( 1.0 + C ) - arctanhroot + arctanfunc ) / u;
    psi[2] = - ( mass + rootC2u / ( 1.0 + C ) - arctanhroot
		 - u * rootC2u / ( 2.0 * ( C + 1.0 ) * um1 )
		 + ( 3.0 * u - 2.0 ) / ( 2.0 * um1 ) * arctanfunc ) /
      ( u * u );
    psi[3] = 2.0 * ( mass + rootC2u / ( 1.0 + C ) - arctanhroot
		     - u * rootC2u / ( 2.0 * ( C + 1.0 ) * um1 )
		     + u * ( C * um1 + u * ( 2.0 + u )
			     - C2 * ( 1.0 + 2.0 * u ) ) /
		     ( 8.0 * ( C + 1.0 ) * rootC2u * um1 * um1 )
		     + ( u * ( 15.0 * u - 20.0 ) + 8.0 ) * arctanfunc /
		     ( 8.0 * um1 * um1 ) ) /
      ( u * u * u );
  }
  else {
    double u_inv = 1.0 / u;
    psi[0] = 0.5 * mass * log ( u );
    psi[1] = mass * u_inv;
    psi[2] = - psi[1] *u_inv;
    psi[3] = - 2.0 * psi[2] * u_inv;
  }

  return;
}

/*--------------------------------------------------------------------------*/

double ample_rho_NFW ( double u )
{
  double r = sqrt ( u );

  return 1.0 / ( 4.0 * M_PI * r * ( 1.0 + r ) * ( 1.0 + r ) );
}

/*--------------------------------------------------------------------------*/

double ample_rho_NFW_trunc ( double u, double C )
{
  double C2 = C * C;
  return ample_rho_NFW ( u ) * C2 / ( C2 + u );
}

/*--------------------------------------------------------------------------*/

double ample_rho_NFW_trunc2 ( double u, double C )
{
  double C2 = C * C;
  double factor = C2 / ( C2 + u );
  return ample_rho_NFW ( u ) * factor * factor;
}

/*--------------------------------------------------------------------------*/ 

double ample_rho_NFW_trunc_hard ( double u, double C )
{
  double C2 = C * C;
  if ( u < C2 )
    return ample_rho_NFW ( u );
  else
    return 0.0;
}

/*--------------------------------------------------------------------------*/

int main ( int argc, char **argv )
{
// The first 3 derivatives are computed
  double psi[4], psiT[4], psiT2[4], psiThard[4], psiS[4];
  double u, C;
  double u_S;
  
// u is a quadratic form computed using axis ratio and orientation angle
// supplied by you in some other function. it is in units of r_s^2 
  u = 100.;

// C is not the concentration, but rather the
// truncation radius in units of the scale radius.
  C = 20.0;

  ample_psi_NFW ( psi, u );
  ample_psi_NFW_trunc ( psiT, u, C );
  ample_psi_NFW_trunc2 ( psiT2, u, C );
  ample_psi_NFW_trunc_hard ( psiThard, u, C );

  u_S = 100.;
  // n = 4 is the de Vaucouleurs profile
  // the Sersic function takes 2n as its last argument
  ample_psi_sersic ( psiS, u_S, 8 );

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "  Test output for ample.c package:\n" );
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "              Scale radius = 1.0\n" );
  fprintf ( stdout, "         Truncation radius = %.1f\n", C);
  fprintf ( stdout, "  Functions evaluated at u = x^2 = %.1f\n", u);
  fprintf ( stdout, "\n" );
  fprintf ( stdout, "               psi        psi'        psi''      psi'''" );
  fprintf ( stdout, "     kappa    (u Gamma)'\n" );
  fprintf ( stdout, "NFW         %.4e %.4e %.4e %.4e  %.4e %.4e\n",
	    psi[0], psi[1], psi[2], psi[3],
	    psi[1] + u * psi[2], psi[2] + u * psi[3] );
  fprintf ( stdout, "NFW (n=1)   %.4e %.4e %.4e %.4e  %.4e %.4e\n",
	    psiT[0], psiT[1], psiT[2], psiT[3],
	    psiT[1] + u * psiT[2], psiT[2] + u * psiT[3] );
  fprintf ( stdout, "NFW (n=2)   %.4e %.4e %.4e %.4e  %.4e %.4e\n",
	    psiT2[0], psiT2[1], psiT2[2], psiT2[3],
	    psiT2[1] + u * psiT2[2], psiT2[2] + u * psiT2[3] );
  fprintf ( stdout, "NFW (hard)  %.4e %.4e %.4e %.4e  %.4e %.4e\n",
	    psiThard[0], psiThard[1], psiThard[2], psiThard[3],
	    psiThard[1] + u * psiThard[2], psiThard[2] + u * psiThard[3] );
  fprintf ( stdout, "Sersic n=4  %.4e %.4e %.4e %.4e  %.4e %.4e\n",
	    psiS[0], psiS[1], psiS[2], psiS[3],
	    psiS[1] + u_S * psiS[2], psiS[2] + u_S * psiS[3] );
  fprintf ( stdout, "\n" );
  
  return 0;
}
/*==========================================================================*/

