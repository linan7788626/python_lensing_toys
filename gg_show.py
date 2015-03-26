#!/usr/bin/env python

import pygame
from pygame.locals import *
from sys import exit
import numpy as np

def xy_rotate(x, y, xcen, ycen, phi):

	phirad = np.deg2rad(phi)
	xnew = (x - xcen) * np.cos(phirad) + (y - ycen) * np.sin(phirad)
	ynew = (y - ycen) * np.cos(phirad) - (x - xcen) * np.sin(phirad)
	return (xnew,ynew)

def gauss_2d(x, y, par):

	(xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
	r_ell_sq = ((xnew**2)*par[4] + (ynew**2)/par[4]) / np.abs(par[1])**2
	return par[0] * np.exp(-0.5*r_ell_sq)

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

def lensed_images_1(xc,yc,nnn):
	boxsize = 4.0
	dsx = boxsize/nnn
	#al1,al2,ka,shi,sh2,mua = lensing_signals_a(kas,aio[0],aio[1],dsx)
	g_amp = 1.0   # peak brightness value
	g_sig = 0.1  # Gaussian "sigma" (i.e., size)
	g_xcen = yc*2.0/nnn  # x position of center
	g_ycen = xc*2.0/nnn  # y position of center
	g_axrat = 1.0 # minor-to-major axis ratio
	g_pa = 0.0	  # major-axis position angle (degrees) c.c.w. from x axis
	gpar = np.asarray([g_amp, g_sig, g_xcen, g_ycen, g_axrat, g_pa])

	xi1 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
	xi2 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
	xi1,xi2 = np.meshgrid(xi1,xi2)

	lpar = np.asarray([0.0,0.0,0.7,0.1,1.0,0.0])
	lpars = np.asarray([0.77,0.7,0.9999999999,0.000000001,0.1,0.0])
	al1,al2,mu = lq_nie(xi1,xi2,lpar)
	al1s,al2s,mus = lq_nie(xi1,xi2,lpars)

	g_image = gauss_2d(xi1,xi2,gpar)

	yi1 = xi1-al1-al1s
	yi2 = xi2-al2-al2s

	g_lensimage = gauss_2d(yi1,yi2,gpar)

	return g_image,g_lensimage

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
	glpars = np.asarray([0.4,0.5*0.1,0.77,0.70,0.999999999,0.0])
	g_lens = gauss_2d(xi1,xi2,glpar)
	g_lens_s = gauss_2d(xi1,xi2,glpars)
	g_lens = g_lens + g_lens_s

	return g_lens,mu,yi1,yi2

def find_critical_curve(mu):
	rows,cols = np.indices(np.shape(mu))
	cdtn = np.sign(mu)*(np.sign(mu[rows-1,cols])+np.sign(mu[rows,cols-1])+np.sign(mu[(rows+1)%len(rows),cols])+np.sign(mu[rows,(cols+1)%len(cols)]))

	res = mu*0
	res[cdtn<4] = 1
	res[cdtn>=4] = 0

	return res

def main():
	nnn = 512
	pygame.init()

	screen = pygame.display.set_mode((nnn, nnn), 0, 32)
	pygame.display.set_caption("Gravitational Lensing Toy!")

	mouse_cursor = pygame.Surface((nnn,nnn))

	base0 = np.zeros((nnn,nnn,3),'uint8')
	base1 = np.zeros((nnn,nnn,3),'uint8')
	base2 = np.zeros((nnn,nnn,3),'uint8')

	g_lens,mu,yi1,yi2 = lensed_images_2(nnn)

	del yi1,yi2,mu

	base0[:,:,0] = g_lens*256
	base0[:,:,1] = g_lens*128
	base0[:,:,2] = g_lens*0

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

		wf = base1+base2

		idx1 = wf>=base0
		idx2 = wf<base0

		base = base0*0
		base[idx1] = wf[idx1]
		base[idx2] = base0[idx2]


		#base = wf*base0+(base1+base2)
		pygame.surfarray.blit_array(mouse_cursor,base)

		screen.blit(mouse_cursor, (0, 0))
		pygame.display.update()

if __name__ == '__main__':
	main()
