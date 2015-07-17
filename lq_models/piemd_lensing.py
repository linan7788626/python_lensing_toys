import numpy as py

def piemd_3d(r,rho0,rcore,rcut):
	res = rho0/((1+r*r/(rcore*rcore))(1+r*r/(rcut*rcut))a)
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

def piemd_kappa(r,sigma0,rcore,rcut,z1,z2):
	res = piemd_sigma(r,sigma0,rcore,rcut)/sigma_cirt(z1,z2)
	return res
#rho(r) = rho0/((1+r*r/a/a)*(1+r*r/s/s))

def piemd_shear(r,rho0,a,s):
	sigma_zero = np.pi*rho0*a*s/(s+a)
	res1 = sigma_zero/sigma_crit(z1,z2)*a*s/(s-a)
	res2 = 2.0*(1.0/(a+np.sqrt(a*a+r*r))-1.0/(s+np.sqrt(s*s+r*r)))
	res3 = 1.0/np.sqrt(a*a+r*r)-1.0/np.sqrt(s*s+r*r)
	res = res1*(res2+res3)
	return res

def piemd_phi(r,rho0,a,s):
	sigma_zero = np.pi*rho0*a*s/(s+a)
	res1 = 4.0*np.pi*G*sigma_zero*a*s/(s-a)
	res2 = np.sqrt(s*s+r*r)-np.sqrt(a*a+r*r)
	res3 = a*np.log(a+np.sqrt(a*a+r*r))-s*np.log(s+np.sqrt(s*s+r*r))
	res = res1*(res2+res3)
	return res

def piemd_alpha(r,rho0,a,s):
	res1 = 8.0*np.pi*G/vc**2*Da2(z1,z2)/Da(z2)*sigma_zero*a*s/(s-a)
	res2 = (r/a)/(1.0+np.sqrt(1+(r/a)**2.0))-(r/s)/(1.0+np.sqrt(1+(r/s)**2.0))
	res = res1*res2
	return res
