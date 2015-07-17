import numpy as np
from scipy import integrate
from mycosmology import *
#----------------------------------------------------------------------------
'''
Black Hole: Point.
Galaxy    : NPL,NIE.
Main halo : GNFW, NFW, PIEMD.
Sub halo  : NFW, NIE, PIEMD.
BCG       : NIE, Burkert, Hernquist, P-Jaffe.

1, 3d density
2, 2d density
3, Deflection angle
3, Deflection potential

1, mass, slope parameters...
2, orientation
3, ellipticity
4, positions
'''
#----------------------------------------------------------------------------
apr =  206269.43		#1/1^{''}
vc = 2.9970e5			#km/s
G = 4.3e-9			#(Mpc/h)^1 (Msun/h)^-1 (km/s)^2
#-------------------------------------------------------------

def UI(x,y,q,alphar):

	def func_UI(u,x,y,q,alphar):
		ksi = np.sqrt(u*(x*x+y*y/(1-(1-q*q)*u)))
		res = ksi/u*alphar(ksi)/np.sqrt(1.0-(1.0-q*q)*u)
		return res

	res = integrate.quad(func_UI, 0, 1.0,args = (x,y,q,alphar))[0]
	return res

def UJ0(x,y,q,kappar):

	def func_UJ0(u,x,y,q,kappar):
		ksi = np.sqrt(u*(x*x+y*y/(1-(1-q*q)*u)))
		res = ksi/u*kappar(ksi)/np.sqrt(1.0-(1.0-q*q)*u)
		return res

	res = integrate.quad(func_UJ0, 0, 1.0,args = (x,y,q,alphar))[0]
	return res

def UJ1(x,y,q,kappar):

	def func_UJ1(u,x,y,q,kappar):
		ksi = np.sqrt(u*(x*x+y*y/(1-(1-q*q)*u)))
		res = ksi/u*kappar(ksi)/(1.0-(1.0-q*q)*u)**1.5
		return res

	res = integrate.quad(func_UJ1, 0, 1.0,args = (x,y,q,kappar))[0]
	return res

def xy_rotate(x, y, xcen, ycen, phi):
	phirad = np.deg2rad(phi)
	xnew = (x - xcen) * np.cos(phirad) + (y - ycen) * np.sin(phirad)
	ynew = (y - ycen) * np.cos(phirad) - (x - xcen) * np.sin(phirad)
	return (xnew,ynew)

def gauss_2d(x,y,n,amp,sig,xc,yc,e,phi):
	(xnew,ynew) = xy_rotate(x, y, xc, yc, phi)
	r_ell_sq = ((xnew)*e + (ynew)/e)/np.abs(sig)**2
	res = amp * np.exp(-0.5*r_ell_sq)
	return res
#-------------------------------------------------------------
'''
Point lens
'''

def pw2(x):
	res = x*x
	return res

def lens_equation_point(x1,x2,re):
	xq = np.sqrt(x1*x1+x2*x2)
	req = re**2.0
	y1 = x1*(1.0-req/xq**2.0)
	y2 = x2*(1.0-req/xq**2.0)
	mu = (1.0-(re/xq)**4.0)**(-1.0)
	return y1,y2,mu
#-------------------------------------------------------------
'''
NFW lens
'''
def func_nfw_bar(x):
	xx = x
	ss = xx*0.0

	idxa = xx>0.0
	idxb = xx<1.0
	idx1 = idxa&idxb
	x1 = 1.0/xx[idx1]**2.0
	x2_up = 2.0*np.arctanh(np.sqrt((1.0-xx[idx1])/(1.0+xx[idx1])))
	x2_down = np.sqrt(1.0-xx[idx1]*xx[idx1])
	x2 = x2_up/x2_down
	x3 = np.log(xx[idx1]/2.0)
	ss[idx1] = x1*(x2+x3)

	idx2 = xx==1.0
	ss[idx2] = 1.0+1.0*np.log(0.5)

	idx3 = xx>1.0
	x1 = 1.0/xx[idx3]**2.0
	x2_up = 2.0*np.arctan(np.sqrt((xx[idx3]-1.0)/(1.0+xx[idx3])))
	x2_down = np.sqrt(xx[idx3]*xx[idx3]-1.0)
	x2 = x2_up/x2_down
	x3 = np.log(xx[idx3]/2.0)
	ss[idx3] = x1*(x2+x3)
	return ss

def lens_equation_nfw(x,ks):
	res = x-4.0*ks*x*func_nfw_bar(x)
	return res

#-------------------------------------------------------------
'''
Burkert lens
'''
def arccoth(x):
	res = 0.5*np.log((x+1)/(x-1))
	return res

def func_burkert(x):

	xx = x
	ss = xx*0.0

	ss1 = np.log(xx/2)+np.pi/4*(np.sqrt(xx*xx+1)-1)+np.sqrt(xx*xx+1)/2*arccoth(np.sqrt(xx*xx+1))

	idxa = xx>0.0
	idxb = xx<1.0
	idx1 = idxa&idxb
	ss[idx1] = ss1[idx1]+0.5*np.sqrt(1-xx[idx1]*xx[idx1])*np.arctanh(np.sqrt(1-xx[idx1]*xx[idx1]))

	idx2 = xx==1.0
	ss[idx2] = -np.log(2.0)-np.pi/4+1.0/(2.0*np.sqrt(2))*(np.pi+np.log(3.0+2.0*np.sqrt(2.0)))

	idx3 = xx>1.0
	ss[idx3] = ss1[idx3]-0.5*np.sqrt(xx[idx3]*xx[idx3]-1)*np.arctan(np.sqrt(xx[idx3]*xx[idx3]-1))

	return ss

def mass_burkert(r,rhos,rs):
	res = 4.0*np.pi*rhos*rs**3.0*func_burkert(r/rs)
	return res

#def lburkert(rhos,rs,z1,z2):
#	res = 16.0*np.pi*G*rhos*rs/vc**2.0*Da(z1)*Da2(z1,z2)/Da(z2)
#	return res
#
#def kappas_burkert(rhos,rs,z1,z2):
#	res = rhos*rs/sigma_crit(z1,z2)
#	return res

def lens_equation_burkert(x,ks):
	res = x-4.0*ks*func_burkert(x)/x
	return res
#-------------------------------------------------------------
'''
NIE lens
'''
def lens_equation_nie(x1,x2,lpar):
	xc1 = lpar[0]
	xc2 = lpar[1]
	q   = lpar[2]
	rc  = lpar[3]
	re  = lpar[4]
	pha = lpar[5]

	phirad = np.deg2rad(pha)
	cosa = np.cos(phirad)
	sina = np.sin(phirad)

	xt1 = (x1-xc1)*cosa+(x2-xc2)*sina
	xt2 = (x2-xc2)*cosa-(x1-xc1)*sina

	phi = np.sqrt(pw2(xt2)+pw2(xt1*q)+pw2(rc))
	sq = np.sqrt(1.0-q*q)
	pd1 = phi+rc/q
	pd2 = phi+rc*q
	fx1 = sq*xt1/pd1
	fx2 = sq*xt2/pd2
	qs = np.sqrt(q)

	a1 = qs/sq*np.arctan(fx1)
	a2 = qs/sq*np.arctanh(fx2)

	xt11 = cosa
	xt22 = cosa
	xt12 = sina
	xt21 =-sina

	fx11 = xt11/pd1-xt1*(xt1*q*q*xt11+xt2*xt21)/(phi*pw2(pd1))
	fx22 = xt22/pd2-xt2*(xt1*q*q*xt12+xt2*xt22)/(phi*pw2(pd2))
	fx12 = xt12/pd1-xt1*(xt1*q*q*xt12+xt2*xt22)/(phi*pw2(pd1))
	fx21 = xt21/pd2-xt2*(xt1*q*q*xt11+xt2*xt21)/(phi*pw2(pd2))

	a11 = qs/(1.0+fx1*fx1)*fx11
	a22 = qs/(1.0-fx2*fx2)*fx22
	a12 = qs/(1.0+fx1*fx1)*fx12
	a21 = qs/(1.0-fx2*fx2)*fx21

	rea11 = (a11*cosa-a21*sina)*re
	rea22 = (a22*cosa+a12*sina)*re
	rea12 = (a12*cosa-a22*sina)*re
	rea21 = (a21*cosa+a11*sina)*re

	y11 = 1.0-rea11
	y22 = 1.0-rea22
	y12 = 0.0-rea12
	y21 = 0.0-rea21

	jacobian = y11*y22-y12*y21
	mu = 1.0/jacobian

	res1 = (a1*cosa-a2*sina)*re
	res2 = (a2*cosa+a1*sina)*re
	return res1,res2,mu
#-------------------------------------------------------------
'''
PIEMD lens
'''
def piemd_3d(r,rho0,rcore,rcut):
	res = rho0/((1+r*r/(rcore*rcore))(1+r*r/(rcut*rcut)))
	return res

def piemd_sigma(r,sigma0,rcore,rcut):
	res1 = sigma0**2.0*rcut/(2.0*G*(rcut-rcore))
	res2 = 1.0/(r*r+rcore*rcore)
	res3 = 1.0/(r*r+rcut*rcut)
	res = res1*(res2-res3)
	return res

def piemd_mass_3d(r,rho0,a,s):
	res1 = 4.0*np.pi*rho0*a*a*s*s/(s*s-a*a)
	res2 = (s*np.arctan2(r,s)-a*np.arctan2(r,a))
	res = res1*res2
	return res

def piemd_mass_2d(r,sigma0,rcore,rcut):
	res1 = np.pi*rcut*sigma0**2.0/G
	res2 = 1.0-(np.sqrt(rcut*rcut+r*r)-np.sqrt(rcore*rcore+r*r))/(rcut-rcore)
	res = res1*res2
	return res

def piemd_mass_total(sigma0,rcore,rcut):
	res = np.pi*sigma0**2.0/G*rcut*rcut/(rcut+rcore)
	return res

def rho0_sigma0(sigma0,rcore,rcut):
	res = sigma0*sigma0/(2.0*np.pi*G)*(rcut+rcore)/(rcore**2.0*rcut)
	return res

def piemd_phi(r,rho0,a,s):
	sigma_zero = np.pi*rho0*a*s/(s+a)
	res1 = 4.0*np.pi*G*sigma_zero*a*s/(s-a)
	res2 = np.sqrt(s*s+r*r)-np.sqrt(a*a+r*r)
	res3 = a*np.log(a+np.sqrt(a*a+r*r))-s*np.log(s+np.sqrt(s*s+r*r))
	res = res1*(res2+res3)
	return res

def piemd_alpha(r,rho0,a,s):
	sigma_zero = np.pi*rho0*a*s/(s+a)
	res1 = 8.0*np.pi*G/vc**2*Da2(z1,z2)/Da(z2)*sigma_zero*a*s/(s-a)
	res2 = (r/a)/(1.0+np.sqrt(1+(r/a)**2.0))-(r/s)/(1.0+np.sqrt(1+(r/s)**2.0))
	res = res1*res2
	return res

def lens_equation_piemd(x,rho0,a,s):
	res = x-piemd_alpha(x,rho0,a,s)
	return res
#-------------------------------------------------------------
'''
Hernquist lens
'''
def sigma2p_hernquist(ksi1,ksi2,a):
	r = np.sqrt(ksi1*ksi1+ksi2*ksi2)
	xp= r/a
	res = r*0.0
	idx1 = np.abs(xp - 1.0) <= 1e-3
	idx2 = np.abs(xp - 1.0) > 1e-3

	x = xp[idx2]
	res1 = 1/(12.0*np.pi*rho_hernquist_2d(x)*a)
	res2 = 0.5/(1-x*x)**3.0
	res3 = -3.0*x*x*xs_hernquist(x)*(8*x**6.0-28*x**4.0+35*x*x-20)-24*x**6+68*x**4-65*x*x+6
	res4 = 6*np.pi*x
	res[idx2] = res1*(res2*res3-res4)

	res[idx1] = 1.0/a*((332-105.0*np.pi)/28.0+(18784.0-5985.0*np.pi)/588.0*(xp[idx1]-1.0) \
			+(707800-225225.0*np.pi)/22638.0*(xp[idx1]-1.0)**2.0)
	return np.sqrt(np.abs(res))

def arcsec(x):
	res = np.arccos(1.0/x)
	return res

def arcsech(x):
	res = np.log((1.0+np.sqrt(1.0-x*x))/x)
	return res

def xs_hernquist(x):
	res = x*0
	idx1 = x>0
	idx2 = x<1
	idx = idx1&idx2
	res[idx] = 1.0/np.sqrt(1-x[idx]*x[idx])*arcsech(x[idx])
	idx = x == 1
	res[idx] = 1.0
	idx = x>1
	res[idx] = 1.0/np.sqrt(x[idx]*x[idx]-1.0)*arcsec(x[idx])
	return res

def sigma2_hernquist_ksi(ksi1,ksi2):
	nz = 400
	zmax = 10.0
	dz = zmax/nz
	ksi = np.sqrt(ksi1*ksi1+ksi2*ksi2)

	sum1 = 0.0
	sum2 = 0.0
	for i in range(nz):
		z = dz * (i+0.5)
		r = np.sqrt(ksi*ksi + z*z)

		sum1 = sum1+rho_hernquist_3d(r)
		sum2 = sum2+rho_hernquist_3d(r)*sigma_hernquist2(r)

	return  np.sqrt(sum2/sum1)

def sigma_hernquist2(r):
	nk=100
	phir = phi_hernquist(r)
	vmax = np.sqrt(2.0*phir)
	v = np.linspace(vmax*0.5/nk,vmax*(nk-0.5)/nk,nk)
	v2_bar = v*0.0
	v4_bar = v*0.0

	e = phir - v*v/2.0
	v2_bar = v**2*fe_hernquist(e)
	v4_bar = v**4*fe_hernquist(e)
	sum1 = np.sum(v2_bar)
	sum2 = np.sum(v4_bar)

	return (1.0/3.0*sum2/sum1)

def fe_hernquist(e):
	res1 = 1/(np.sqrt(2.0)*(2*np.pi)**3)
	res2 = np.sqrt(e)/(1.0-e)**2
	res3 = (1-2*e)*(8*e*e-8*e-3)
	res4 = 3*np.arcsin(np.sqrt(e))/np.sqrt(e*(1-e))
	res = res1*res2*(res3+res4)
	return res

def rho_hernquist_3d(r):
	return 1.0/((2.0*np.pi)*r*(1+r)**3)

def rho_hernquist_2d(x):
	res1 = 1.0/(2.0*np.pi*(1-x*x)**2.0)
	res2 = (2.0+x*x)*xs_hernquist(x)-3.0
	res = res1*res2
	return res

def phi_hernquist(r):
	return 1/(1+r)

def func_hernquist(x):
	res = x*0
	idx1 = x>0
	idx2 = x<1
	idx = idx1&idx2
	res[idx] = 1.0/np.sqrt(1-x[idx]*x[idx])*np.arctanh(np.sqrt(1-x[idx]*x[idx]))
	idx = x == 1
	res[idx] = 1.0
	idx = x>1
	res[idx] = 1.0/np.sqrt(x[idx]*x[idx]-1.0)*np.arctanh(np.sqrt(x[idx]*x[idx]-1))
	return res
#x = r/rs ks = rhos*rs/sigma_crit
def lens_equation_hernquist(x,rs,ks):
	res = 2.0*ks*rs*x*(1-func_hernquist(x))/(x*x-1)
	return res

#-------------------------------------------------------------
'''
P-Jaffe lens
'''
def kappa_pjaffe(x,s,a,b):
	res = b/2*(1/np.sqrt(s*s+x*x)-1/np.sqrt(a*a+x*x))
	return res
#-------------------------------------------------------------
'''
GNFW lens
'''
