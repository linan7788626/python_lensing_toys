import numpy as np
from matplotlib.pylab import *
from alens_arr import *
from mycosmology import *
##-------------------------------------------------------------
#rr,zz = np.loadtxt('sigma.dat', usecols=(0,1),unpack=True)
#xx = 10.0**np.linspace(-2,1,99)
#yy = sigma2p(xx,0.0,5.0)
#plot(xx,yy,'b-',lw=2)
##plot(xx,yy2,'ro',lw=2)
#plot(rr,zz,'g-',lw=2)
#show()

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
def shear_point_mass(m,x1,x2,z1,z2):
	res = m/((np.pi)*sigma_crit(z1,z2))/(x1*x1+x2*x2)
	return res
def shear_pm(x1,x2,re):
	ree = re/apr*Da(0.1)
	res = (ree*ree)/(x1*x1+x2*x2)
	return res

mtotal = 1e14
zl = 0.1
zs = 1.0
re0 = re_m_all(mtotal,zl,zs)
#print (re0*Da(0.1)/apr)**2.0,1e14/np.pi/sigma_crit(zl,zs)

xx = np.linspace(1,5,100)
shp1 = xx*0.0
shp2 = xx*0.0
shp1 = shear_point_mass(1e14,0.0,xx,zl,zs)
shp2 = shear_pm(0.0,xx,re0)
for i in range(100):
	print xx[i],shp1[i]
