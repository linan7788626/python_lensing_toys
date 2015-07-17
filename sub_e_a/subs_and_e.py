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
	return res1*re,res2*re,jacobian
#----------------------------------------------------------------------
def le_sie_subs(x1,x2,lpar1,lpar2):
	rea  = lpar1[4]
	xac1 = lpar1[0]
	xac2 = lpar1[1]
	qa   = lpar1[2]
	rca  = lpar1[3]/rea
	pha  = lpar1[5]

	reb  = lpar2[4]
	xbc1 = lpar2[0]
	xbc2 = lpar2[1]
	qb   = lpar2[2]
	rcb  = lpar2[3]/reb
	phb  = lpar2[5]
#------------------------------------halo a-----------------------------
	phirada = np.deg2rad(pha)
	xta1 = ((x1-xac1)*np.cos(phirada)+(x2-xac2)*np.sin(phirada))/rea
	xta2 = ((x2-xac2)*np.cos(phirada)-(x1-xac1)*np.sin(phirada))/rea

	phia = np.sqrt(xta2*xta2+xta1*xta1*qa*qa+rca*rca*qa*qa)
	sqa = np.sqrt(1.0-qa*qa)
	fxa1 = sqa*xta1/(phia+rca)
	fxa2 = sqa*xta2/(phia+rca*qa*qa)
	qsa = np.sqrt(qa)

	aa1 = -qsa/sqa*np.arctan(fxa1)
	aa2 = -qsa/sqa*np.arctanh(fxa2)
#--------------------------------halo b---------------------------------
	phiradb = np.deg2rad(phb)
	xtb1 = ((x1-xbc1)*np.cos(phiradb)+(x2-xbc2)*np.sin(phiradb))/reb
	xtb2 = ((x2-xbc2)*np.cos(phiradb)-(x1-xbc1)*np.sin(phiradb))/reb

	phib = np.sqrt(xtb2*xtb2+xtb1*xtb1*qb*qb+rcb*rcb*qb*qb)
	sqb = np.sqrt(1.0-qb*qb)
	fxb1 = sqb*xtb1/(phib+rcb)
	fxb2 = sqb*xtb2/(phib+rcb*qb*qb)
	qsb = np.sqrt(qb)

	ab1 = -qsb/sqb*np.arctan(fxb1)
	ab2 = -qsb/sqb*np.arctanh(fxb2)
#----------------------------------------------------------------------
	resa1 = (aa1)*np.cos(phirada)-(aa2)*np.sin(phirada)
	resa2 = (aa2)*np.cos(phirada)+(aa1)*np.sin(phirada)

	resb1 = (ab1)*np.cos(phiradb)-(ab2)*np.sin(phiradb)
	resb2 = (ab2)*np.cos(phiradb)+(ab1)*np.sin(phiradb)

	res1 = resa1*rea+resb1*reb
	res2 = resa2*rea+resb2*reb
#----------------------------------------------------------------------
	dphia_x1 = xta1*qa*qa/phia
	dphia_x2 = xta2/phia
	dfx1a_x1 = sqa*(1.0/(phia+rca)-xta1/pw2(phia+rca)*dphia_x1)
	dfx2a_x2 = sqa*(1.0/(phia+rca*qa*qa)-xta2/pw2(phia+rca*qa*qa)*dphia_x2)
	dfx1a_x2 = sqa*xta1*dphia_x2/pw2(phia+rca)
	dfx2a_x1 = sqa*xta2*dphia_x1/pw2(phia+rca*qa*qa)

	dphib_x1 = xtb1*qb*qb/phib
	dphib_x2 = xtb2/phib
	dfx1b_x1 = sqb*(1.0/(phib+rcb)-xtb1/pw2(phib+rcb)*dphib_x1)
	dfx2b_x2 = sqb*(1.0/(phib+rcb*qb*qb)-xtb2/pw2(phib+rcb*qb*qb)*dphib_x2)
	dfx1b_x2 = sqb*xtb1*dphib_x2/pw2(phib+rcb)
	dfx2b_x1 = sqb*xtb2*dphib_x1/pw2(phib+rcb*qb*qb)

	#psy1_x1 = 1.0-qsa/(sqa*(1.0+fxa1*fxa1))*dfx1a_x1 \
	#		-qsb/(sqb*(1.0+fxb1*fxb1))*dfx1b_x1
	#psy2_x2 = 1.0-qsa/(sqa*(1.0-fxa2*fxa2))*dfx2a_x2 \
	#		-qsb/(sqb*(1.0-fxb2*fxb2))*dfx2b_x2
	#psy1_x2 = qsa/(sqa*(1.0+fxa1*fxa1))*dfx1a_x2 \
	#		+qsb/(sqb*(1.0+fxb1*fxb1))*dfx1b_x2
	#psy2_x1 = qsa/(sqa*(1.0-fxa2*fxa2))*dfx2a_x1 \
	#		+qsb/(sqb*(1.0-fxb2*fxb2))*dfx2b_x1

	#psy1_x1 = 1.0-qsa/(sqa*(1.0+fxa1*fxa1))*dfx1a_x1
	#psy2_x2 = 1.0-qsa/(sqa*(1.0-fxa2*fxa2))*dfx2a_x2
	#psy1_x2 = qsa/(sqa*(1.0+fxa1*fxa1))*dfx1a_x2
	#psy2_x1 = qsa/(sqa*(1.0-fxa2*fxa2))*dfx2a_x1

	y11p1 = qsa/(sqa*(1.0+fxa1*fxa1))*dfx1a_x1
	y11p2 = qsb/(sqb*(1.0+fxb1*fxb1))*dfx1b_x1
	y22p1 = qsa/(sqa*(1.0-fxa2*fxa2))*dfx2a_x2
	y22p2 = qsb/(sqb*(1.0-fxb2*fxb2))*dfx2b_x2
	y12p1 = qsa/(sqa*(1.0+fxa1*fxa1))*dfx1a_x2
	y12p2 = qsb/(sqb*(1.0+fxb1*fxb1))*dfx1b_x2
	y21p1 = qsa/(sqa*(1.0-fxa2*fxa2))*dfx2a_x1
	y21p2 = qsb/(sqb*(1.0-fxb2*fxb2))*dfx2b_x1


	y11 = 1.0-y11p1-y11p2
	y22 = 1.0-y22p1-y22p2
	y12 = y12p1+y12p2
	y21 = y21p1+y21p2
	jacobian = y11*y22-y12*y21
	mu = 1.0/jacobian

	return res1,res2,mu
#----------------------------------------------------------------------
g_amp = 1.0   # peak brightness value
g_sig = 0.05  # Gaussian "sigma" (i.e., size)
g_xcen = 0.0  # x position of center
g_ycen = 0.0  # y position of center
g_axrat = 1.0 # minor-to-major axis ratio
g_pa = 0.0    # major-axis position angle (degrees) c.c.w. from x axis
gpar = np.asarray([g_amp, g_sig, g_xcen, g_ycen, g_axrat, g_pa])
##----------------------------------------------------------------------
l_xcen = 0.5
l_ycen = 0.5
l_q = 0.99999999999
l_rc = 0.00000000001
l_re = 1.00000000001
l_pa = 0.0
l_par = np.asarray([l_xcen, l_ycen, l_q, l_rc,l_re,l_pa])
#----------------------------------------------------------------------
l_xcens = -0.5
l_ycens = -0.5
l_qs = 0.99999999999
l_rcs = 0.00000000001
l_res = 0.00000000001
l_pas = 0.0
l_pars = np.asarray([l_xcens, l_ycens, l_qs, l_rcs,l_res,l_pas])
#----------------------------------------------------------------------
boxsize = 5.0
nn = 512

ri = np.linspace(0.0,boxsize/2.0,nn)
ti = np.linspace(0.0,2.0*np.pi,nn)
ri,ti = np.meshgrid(ri,ti)

xi1 = ri*np.cos(ti)
xi2 = ri*np.sin(ti)

g_image = ldf.gauss_2d(xi1, xi2, gpar)
(ai1, ai2, mui) = le_sie_subs(xi1,xi2,l_par,l_pars)

yi1 = xi1+ai1
yi2 = xi2+ai2
g_lensimage = ldf.gauss_2d(yi1, yi2, gpar)
#--------------------------lens images contour------------------------
levels = [0.15,0.30,0.45,0.60,0.75,0.9,1.05]
lev2 = [1000]
figure(num=None,figsize=(10,5),dpi=80, facecolor='w', edgecolor='k')


a = axes([0.05,0.1,0.4,0.8])
a.set_xlim(-2.5,2.5)
a.set_ylim(-2.5,2.5)
a.contourf(xi1,xi2,g_image,levels)
a.contour(yi1,yi2,np.abs(mui),lev2,colors=('g'),linewidths = 2.0)

b = axes([0.55,0.1,0.4,0.8])
b.set_xlim(-2.5,2.5)
b.set_ylim(-2.5,2.5)
b.contourf(xi1,xi2,g_lensimage,levels)
b.contour(xi1,xi2,np.abs(mui),lev2,colors=('k'),linewidths = 2.0)
savefig('output.eps')
show()
#------------------------------------------------------
#from mpl_toolkits.mplot3d import axes3d
#fig = figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(xi1, xi2, mui, rstride=10, cstride=10)
##ax.plot_surface(xi1, xi2, mui, rstride=10, cstride=10)
#ax.set_zlim3d(-40,40)
#plt.show()
#colorbar(r
