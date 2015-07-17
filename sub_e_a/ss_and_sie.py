import numpy as np
from matplotlib.pylab import *
import lensdemo_funcs as ldf
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
def lens_equation_sie(x1,x2,lpar):
	re  = lpar[4]
	x1  = x1/re
	x2  = x2/re
	xc1 = lpar[0]/re
	xc2 = lpar[1]/re
	q   = lpar[2]
	rc  = lpar[3]/re
	phi = lpar[5]

	phirad = np.deg2rad(phi)
	xt1 = ((x1-xc1)*np.cos(phirad)+(x2-xc2)*np.sin(phirad))
	xt2 = ((x2-xc2)*np.cos(phirad)-(x1-xc1)*np.sin(phirad))

	phi = np.sqrt(xt2*xt2+xt1*xt1*q*q+rc*rc*q*q)
	sq = np.sqrt(1.0-q*q)
	fx1 = sq*xt1/(phi+rc)
	fx2 = sq*xt2/(phi+rc*q*q)
	qs = np.sqrt(q)


	y1 = xt1-qs/sq*np.arctan(fx1)
	y2 = xt2-qs/sq*np.arctanh(fx2)

	dphi_x1 = xt1*q*q/phi
	dphi_x2 = xt2/phi
	dfx1_x1 = sq*(1.0/(phi+rc)-xt1/pw2(phi+rc)*dphi_x1)
	dfx2_x2 = sq*(1.0/(phi+rc*q*q) -xt2/pw2(phi+rc*q*q)*dphi_x2)
	dfx1_x2 = sq*xt1*dphi_x2/pw2(phi+rc)
	dfx2_x1 = sq*xt2*dphi_x1/pw2(phi+rc*q*q)

	psy1_x1 = 1.0-qs/sq/(1.0+fx1*fx1)*dfx1_x1
	psy2_x2 = 1.0-qs/sq/(1.0-fx2*fx2)*dfx2_x2
	psy1_x2 = qs/(sq*(1.0+fx1*fx1))*dfx1_x2
	psy2_x1 = qs/(sq*(1.0-fx2*fx2))*dfx2_x1

	jacobian = psy1_x1*psy2_x2-psy1_x2*psy2_x1
	mu = 1.0/(psy1_x1*psy2_x2-psy1_x2*psy2_x1)

	res1 = (y1-xt1)*np.cos(phirad) - (y2-xt2)*np.sin(phirad)
	res2 = (y2-xt2)*np.cos(phirad) + (y1-xt1)*np.sin(phirad)
	return res1*re,res2*re,mu
##----------------------------------------------------------------------
g_amp = 1.0   # peak brightness value
g_sig = 0.05  # Gaussian "sigma" (i.e., size)
g_xcen = 0.0  # x position of center
g_ycen = 0.0  # y position of center
g_axrat = 1.0 # minor-to-major axis ratio
g_pa = 0.0    # major-axis position angle (degrees) c.c.w. from x axis
gpar = np.asarray([g_amp, g_sig, g_xcen, g_ycen, g_axrat, g_pa])
##----------------------------------------------------------------------
l_xcen = 0.0
l_ycen = 0.0
l_q = 0.7
l_rc = 0.1
l_re = 1.0
l_pa = 0.0
l_par = np.asarray([l_xcen, l_ycen, l_q, l_rc,l_re,l_pa])
#----------------------------------------------------------------------
l_xcens = 0.57
l_ycens = 0.61
l_qs = 0.9999999
l_rcs = 0.00000001
l_res = 0.1
l_pas = 0.0
l_pars = np.asarray([l_xcens, l_ycens, l_qs, l_rcs,l_res,l_pas])
#----------------------------------------------------------------------
boxsize = 5.0
nn = 512

xi1 = (np.linspace(-nn/2.0,nn/2.0,nn-1)+0.5)*boxsize/nn
xi2 = (np.linspace(-nn/2.0,nn/2.0,nn-1)+0.5)*boxsize/nn
xi1,xi2 = np.meshgrid(xi1,xi2)

#ri = np.linspace(0.0,boxsize/2.0,nn)
#ti = np.linspace(0.0,2.0*np.pi,nn)
#ri,ti = np.meshgrid(ri,ti)
#
#xi1 = ri*np.cos(ti)
#xi2 = ri*np.sin(ti)

g_image = ldf.gauss_2d(xi1, xi2, gpar)
(ai1, ai2, mui) = lens_equation_sie(xi1,xi2,l_par)
(aj1, aj2, muj) = lens_equation_sie(xi1,xi2,l_pars)

yi1 = xi1+ai1+aj1
yi2 = xi2+ai2+aj2
mui = mui+muj

g_lensimage = ldf.gauss_2d(yi1, yi2, gpar)
#--------------------------lens images contour------------------------
levels = [0.15,0.30,0.45,0.60,0.75,0.9,1.05]
figure(num=None,figsize=(10,5),dpi=80, facecolor='w', edgecolor='k')


a = axes([0.05,0.1,0.4,0.8])
a#.set_xlim(-2.5,2.5)
a.set_ylim(-2.5,2.5)
a.contourf(xi1,xi2,g_image,levels)
a.contour(yi1,yi2,mui,colors=('g'),linewidths = 2.0)

b = axes([0.55,0.1,0.4,0.8])
b.set_xlim(-2.5,2.5)
b.set_ylim(-2.5,2.5)
b.contourf(xi1,xi2,g_lensimage,levels)
b.contour(xi1,xi2,mui,colors=('k'),linewidths = 2.0)
savefig('output.eps')
show()
#contourf(xg1,xg2,g_lensimagej)
#colorbar()
