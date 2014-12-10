#!/usr/bin/env python

import pygame
from pygame.locals import *
from sys import exit
from pylab import *
import numpy as np

import ctypes as ct
fcic = ct.CDLL("./libfcic.so")
fcic.forward_cic.argtypes = [np.ctypeslib.ndpointer(dtype =  ct.c_double),\
							 np.ctypeslib.ndpointer(dtype =  ct.c_double), \
							 np.ctypeslib.ndpointer(dtype =  ct.c_double), \
							 ct.c_double,ct.c_double,ct.c_int,ct.c_int,ct.c_int,\
							 np.ctypeslib.ndpointer(dtype = ct.c_double)]
fcic.forward_cic.restype  = ct.c_void_p

def call_forward_cic(nx1,nx2,boxsize,yif1,yif2):
	img_in = np.array(np.ones(len(yif1)),dtype=ct.c_double)
	yif1 = np.array(yif1,dtype=ct.c_double)
	yif2 = np.array(yif2,dtype=ct.c_double)
	img_out = np.zeros((nx1,nx2))
	fcic.forward_cic(img_in,yif1,yif2,ct.c_double(boxsize),ct.c_double(boxsize),ct.c_int(nx1),ct.c_int(nx2),ct.c_int(len(yif1)),img_out)
	return img_out.T

def xy_rotate(x, y, xcen, ycen, phi):

	phirad = np.deg2rad(phi)
	xnew = (x - xcen) * np.cos(phirad) + (y - ycen) * np.sin(phirad)
	ynew = (y - ycen) * np.cos(phirad) - (x - xcen) * np.sin(phirad)
	return (xnew,ynew)

def gauss_2d(x, y, par):

	(xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
	r_ell_sq = ((xnew**2)*par[4] + (ynew**2)/par[4]) / np.abs(par[1])**2
	return par[0] * np.exp(-0.5*r_ell_sq)

#-------------------------------------------------------------
def lq_nie(x1,x2,lpar):
	xc1 = lpar[0]
	xc2 = lpar[1]
	q	= lpar[2]
	rc	= lpar[3]
	re	= lpar[4]
	pha = lpar[5]

	phirad = np.deg2rad(pha)
	cosa = np.cos(phirad)
	sina = np.sin(phirad)

	xt1 = (x1-xc1)*cosa+(x2-xc2)*sina
	xt2 = (x2-xc2)*cosa-(x1-xc1)*sina

	phi = np.sqrt(xt2*xt2+xt1*q*xt1*q+rc*rc)
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

	fx11 = xt11/pd1-xt1*(xt1*q*q*xt11+xt2*xt21)/(phi*pd1*pd1)
	fx22 = xt22/pd2-xt2*(xt1*q*q*xt12+xt2*xt22)/(phi*pd2*pd2)
	fx12 = xt12/pd1-xt1*(xt1*q*q*xt12+xt2*xt22)/(phi*pd1*pd1)
	fx21 = xt21/pd2-xt2*(xt1*q*q*xt11+xt2*xt21)/(phi*pd2*pd2)

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


 #---------------------------------------------------------------------------------
import scipy.ndimage.interpolation as sn
def mapping_on_plane(img_in,xi1,xi2,dsi):
	nsx, nsy = np.shape(img_in)
	xt1 = (xi1)/dsi+nsx/2.0
	xt2 = (xi2)/dsi+nsy/2.0
	points = np.array([np.array(xt1.flat),np.array(xt2.flat)])
	img_out = sn.map_coordinates(img_in,points,order=1,mode='constant')
	return img_out.reshape((nsx,nsy))
#--------------------------------------------------------------------
def lensed_images_1(xc,yc,nnn):
	boxsize = 4.0
	dsx = boxsize/nnn
	#al1,al2,ka,shi,sh2,mua = lensing_signals_a(kas,aio[0],aio[1],dsx)
	g_amp = 1.0   # peak brightness value
	g_sig = 0.02  # Gaussian "sigma" (i.e., size)
	g_xcen = yc*2.0/nnn  # x position of center
	g_ycen = xc*2.0/nnn  # y position of center
	g_axrat = 1.0 # minor-to-major axis ratio
	g_pa = 0.0	  # major-axis position angle (degrees) c.c.w. from x axis
	gpar = np.asarray([g_amp, g_sig, g_xcen, g_ycen, g_axrat, g_pa])
#----------------------------------------------------------------------

	xi1 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
	xi2 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
	xi1,xi2 = np.meshgrid(xi1,xi2)

	lpar = np.asarray([0.0,0.0,0.7,0.1,1.0,0.0])
	al1,al2,mu = lq_nie(xi1,xi2,lpar)

	g_image = gauss_2d(xi1,xi2,gpar)

	yi1 = xi1-al1
	yi2 = xi2-al2

	g_lensimage = gauss_2d(yi1,yi2,gpar)

	return g_image,g_lensimage
#--------------------------------------------------------------------
def lensed_images_2(nnn):
	boxsize = 4.0
	dsx = boxsize/nnn

	xi1 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
	xi2 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
	xi1,xi2 = np.meshgrid(xi1,xi2)

	lpar = np.asarray([0.0,0.0,0.7,0.1,1.0,0.0])
	al1,al2,mu = lq_nie(xi1,xi2,lpar)
	yi1 =xi1-al1
	yi2 =xi2-al2

	glpar = np.asarray([1.0,0.5,0.0,0.0,0.7,0.0])
	g_lens = gauss_2d(xi1,xi2,glpar)

	return g_lens,mu,yi1,yi2
#--------------------------------------------------------------------
def source_plane_finer(xi1,xi2):

	lpar = np.asarray([0.0,0.0,0.7,0.1,1.0,0.0])
	al1,al2,mu = lq_nie(xi1,xi2,lpar)
	yi1 =xi1-al1
	yi2 =xi2-al2

	return mu,yi1,yi2

def find_critical_curve(mu):
	rows,cols = np.indices(np.shape(mu))
	cdtn = np.sign(mu)*(np.sign(mu[rows-1,cols])+np.sign(mu[rows,cols-1])+np.sign(mu[(rows+1)%len(rows),cols])+np.sign(mu[rows,(cols+1)%len(cols)]))

	res = mu*0
	res[cdtn<4] = 1
	res[cdtn>=4] = 0

	return res

import scipy.interpolate as si

def mapping_on_kth_plane(img_in,xi1,xi2,yi1,yi2,yif1,yif2,muf):
	ya1 = np.hstack([yi1.flat,yif1.flat])
	ya2 = np.hstack([yi2.flat,yif2.flat])
	mua = np.hstack([img_in.flat,muf.flat])
	#ya1 = np.array(yif1.flat)
	#ya2 = np.array(yif2.flat)
	#mua = np.array(muf.flat)
	points = np.array([ya1,ya2])
	res = si.griddata(points.T, mua, (xi1,xi2), method='linear')
	return res

def refine_critical(critical,xi1,xi2,dsx,nfiner=8):
	x1tmp0 = xi1[critical>0]
	yift1 = np.zeros((len(x1tmp0),nfiner,nfiner))
	yift2 = np.zeros((len(x1tmp0),nfiner,nfiner))
	muf = np.zeros((len(x1tmp0),nfiner,nfiner))
	dsf = dsx/nfiner/2
	for i in xrange(nfiner):
		for j in xrange(nfiner):
			x1tmp = xi1[critical>0]+(dsf*(1-nfiner)*0.5)+dsf*i
			x2tmp = xi2[critical>0]+(dsf*(1-nfiner)*0.5)+dsf*j

			muf[:,i,j],yift1[:,i,j],yift2[:,i,j] = source_plane_finer(x1tmp,x2tmp)

	return np.ones((len(x1tmp0),nfiner,nfiner)),yift1,yift2

def build_polar_coor(r,phi):
	x1 = r*cos(phi)
	x2 = r*sin(phi)
	return x1,x2

def main():
	boxsize = 4.0
	nnn = 512
	dsx = boxsize/nnn

	xi1 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
	xi2 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
	xi1,xi2 = np.meshgrid(xi1,xi2)


	pygame.init()

	screen = pygame.display.set_mode((nnn, nnn), 0, 32)
	pygame.display.set_caption("Gravitational Lensing Toy!")

	mouse_cursor = pygame.Surface((nnn,nnn))

	base0 = np.zeros((nnn,nnn,3),'uint8')
	base1 = np.zeros((nnn,nnn,3),'uint8')
	base2 = np.zeros((nnn,nnn,3),'uint8')
	base3 = np.zeros((nnn,nnn,3),'uint8')
	base4 = np.zeros((nnn,nnn,3),'uint8')

	g_lens,mu,yi1,yi2 = lensed_images_2(nnn)
	del g_lens

	critical = find_critical_curve(mu)
	base3[:,:,0] = critical*255
	base3[:,:,1] = critical*0
	base3[:,:,2] = critical*0

	muf,yif1,yif2 = refine_critical(critical,xi1,xi2,dsx)

	caustic = call_forward_cic(nnn,nnn,boxsize,yif1.flat,yif2.flat)

	caustic[caustic>0]=1

	base4[:,:,0] = caustic*0
	base4[:,:,1] = caustic*255
	base4[:,:,2] = caustic*0

	del xi1,xi2,yi1,yi2,yif1,yif2,muf

	while True:
		for event in pygame.event.get():
			if event.type == QUIT:
				exit()

		x, y = pygame.mouse.get_pos()
		x-= mouse_cursor.get_width() / 2
		y-= mouse_cursor.get_height() / 2

		g_image,g_lensimage = lensed_images_1(x,y,nnn)

		base1[:,:,0] = g_image*256
		base1[:,:,1] = g_image*256
		base1[:,:,2] = g_image*256

		base2[:,:,0] = g_lensimage*102
		base2[:,:,1] = g_lensimage*178
		base2[:,:,2] = g_lensimage*256

		base = base3+base4+(base1+base2)
		pygame.surfarray.blit_array(mouse_cursor,base)

		screen.blit(mouse_cursor, (0, 0))
		pygame.display.update()

if __name__ == '__main__':
	main()
