import numpy as np

def sigma2p(ksi1,ksi2,a):
	r = np.sqrt(ksi1*ksi1+ksi2*ksi2)
	xp= r/a
	res = r*0.0
	idx1 = np.abs(xp - 1.0) <= 1e-3
	idx2 = np.abs(xp - 1.0) > 1e-3

	x = xp[idx2]
	res1 = 1/(12.0*np.pi*rho_2d(x)*a)
	res2 = 0.5/(1-x*x)**3.0
	res3 = -3.0*x*x*xs(x)*(8*x**6.0-28*x**4.0+35*x*x-20)-24*x**6+68*x**4-65*x*x+6
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
def xs(x):
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

def sigma2_ksi(ksi1,ksi2):
	nz = 400
	zmax = 10.0
	dz = zmax/nz
	ksi = np.sqrt(ksi1*ksi1+ksi2*ksi2)
	
	sum1 = 0.0
	sum2 = 0.0
	for i in range(nz):
		z = dz * (i+0.5)
		r = np.sqrt(ksi*ksi + z*z)
		
		sum1 = sum1+rho(r)
		sum2 = sum2+rho(r)*sigma2(r)
	
	return  np.sqrt(sum2/sum1)

#/* velocity dispersion squared as a function of radius */
def gauss2p(x, y, sig):
    r_ell_sq = ((x**2) + (y**2)) / np.abs(sig)**2
    return np.exp(-0.5*r_ell_sq)
def sigma2(r):
	nk=100
	phir = phi(r)
	vmax = np.sqrt(2.0*phir)
	v = np.linspace(vmax*0.5/nk,vmax*(nk-0.5)/nk,nk)
	v2_bar = v*0.0
	v4_bar = v*0.0
	
	e = phir - v*v/2.0
	v2_bar = v**2*fe(e)
	v4_bar = v**4*fe(e)
	sum1 = np.sum(v2_bar)
	sum2 = np.sum(v4_bar)

	return (1.0/3.0*sum2/sum1)


#/* Ergodic distribution */
def fe(e):
	res1 = 1/(np.sqrt(2.0)*(2*np.pi)**3)
	res2 = np.sqrt(e)/(1.0-e)**2
	res3 = (1-2*e)*(8*e*e-8*e-3)
	res4 = 3*np.arcsin(np.sqrt(e))/np.sqrt(e*(1-e))
	res = res1*res2*(res3+res4)
	return res

def rho(r):
	return 1.0/((2.0*np.pi)*r*(1+r)**3)
def rho_2d(x):
	res1 = 1.0/(2.0*np.pi*(1-x*x)**2.0)
	res2 = (2.0+x*x)*xs(x)-3.0
	res = res1*res2
	return res


def phi(r):
	return 1/(1+r)
