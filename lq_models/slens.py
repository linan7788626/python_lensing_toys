import numpy as np
from mycosmology import *
import alens_arr as aa
#-------------------------------------------------------------

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
def lens_equation_nfw(x,ks):
	res = x-4.0*ks*x*aa.func_bar(x)
	return res

#-------------------------------------------------------------
def arccoth(x):
	res = 0.5*np.log((x+1)/(x-1))
	return res
#-------------------------------------------------------------
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

def lburkert(rhos,rs,z1,z2):
	res = 16.0*np.pi*G*rhos*rs/vc**2.0*Da(z1)*Da2(z1,z2)/Da(z2)
	return res

def kappas(rhos,rs,z1,z2):
	res = rhos*rs/sigma_crit(z1,z2)
	return res


def lens_equation_burkert(x,ks):
	#rho(r) = rhos/(1+r/rs)/(1+(r/rs)**2)
	res = x-4.0*ks*func_burkert(x)/x
	return res

#from pylab import *
#
#xlim(-2,2)
#ylim(-0.6,0.6)
#xt = np.linspace(-2.0,2.0,100)
#idx = xt<=0
#yt1 = lens_equation_burkert(xt,2.0/4.0)
#yt2 = lens_equation_burkert(xt,8.0/np.pi/4.0)
#yt3 = lens_equation_burkert(xt,4.0/4.0)
#yt1[idx] = -lens_equation_burkert(np.abs(xt[idx]),2.0/4.0)
#yt2[idx] = -lens_equation_burkert(np.abs(xt[idx]),8.0/np.pi/4.0)
#yt3[idx] = -lens_equation_burkert(np.abs(xt[idx]),4.0/4.0)
#
#plot(xt,yt1,'r.')
#plot(xt,yt2,'r--')
#plot(xt,yt3,'r-')
#show()

from pylab import *

#xlim(0,2)
#ylim(-0.6,0.6)
ks0 = 1.2
xt = np.linspace(0.0001,8.0,100)
yt1 = xt - lens_equation_burkert(xt,ks0)
yt2 = xt - lens_equation_nfw(xt,ks0)

plot(xt,yt1,'r-',label="Burkert")
plot(xt,yt2,'b-',label="NFW")
legend()
show()

#-------------------------------------------------------------
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
