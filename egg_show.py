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
    q   = lpar[2]
    rc  = lpar[3]
    re  = lpar[4]
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

def lensed_images(xi1,xi2,gpar,lpar,lpars):

    al1,al2,mu = lq_nie(xi1,xi2,lpar)
    al1s,al2s,mus = lq_nie(xi1,xi2,lpars)

    g_image = gauss_2d(xi1,xi2,gpar)

    yi1 = xi1-al1-al1s
    yi2 = xi2-al2-al2s

    g_lensimage = gauss_2d(yi1,yi2,gpar)

    return g_image,g_lensimage

def lens_galaxies(xi1,xi2,glpar,glpars):

    g_lens = gauss_2d(xi1,xi2,glpar)
    g_lens_s = gauss_2d(xi1,xi2,glpars)
    g_lens = g_lens# + g_lens_s

    return g_lens

def find_critical_curve(mu):
    rows,cols = np.indices(np.shape(mu))
    cdtn = np.sign(mu)*(np.sign(mu[rows-1,cols])+np.sign(mu[rows,cols-1])+np.sign(mu[(rows+1)%len(rows),cols])+np.sign(mu[rows,(cols+1)%len(cols)]))

    res = mu*0
    res[cdtn<4] = 1
    res[cdtn>=4] = 0

    return res

def keyPressed(inputKey):
    keysPressed = pygame.key.get_pressed()
    if keysPressed[inputKey]:
        return True
    else:
        return False


def main():


    nnn = 512
    boxsize = 4.0
    dsx = boxsize/nnn
    xi1 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
    xi2 = np.linspace(-boxsize/2.0,boxsize/2.0-dsx,nnn)+0.5*dsx
    xi1,xi2 = np.meshgrid(xi1,xi2)

    pygame.init()
    FPS = 30
    fpsClock = pygame.time.Clock()

    screen = pygame.display.set_mode((nnn, nnn), 0, 32)

    pygame.display.set_caption("Gravitational Lensing Toy")

    mouse_cursor = pygame.Surface((nnn,nnn))

    #----------------------------------------------------
    # lens parameters for subhalo
    xlcs = 0.7
    ylcs = 0.77
    qls = 0.999999999
    rcs = 0.000000001
    res = 0.0
    phis = 0.0

    #----------------------------------------------------
    # parameters of NIE model (lens model, deflection angles)
    #----------------------------------------------------
    # 1, y position of center
    # 2, x position of center
    # 3, minor-to-major axis ratio
    # 4, size of flat core
    # 5, Einstein radius (lensing strength)
    # 6, major-axis position angle (degrees) c.c.w. from y axis
    lpars = np.asarray([ylcs,xlcs,qls,rcs,res,phis])


    #----------------------------------------------------
    # luminosity parameters for subhalo
    aps = 0.4
    l_sigs = 0.05
    #----------------------------------------------------
    # Parameters of Gaussian model (luminosity distribution of lenses)
    #----------------------------------------------------
    # 1, peak brightness value
    # 2, Gaussian "sigma" (i.e., size)
    # 3, y position of center
    # 4, x position of center
    # 5, minor-to-major axis ratio
    # 6, major-axis position angle (degrees) c.c.w. from y axis

    glpars = np.asarray([aps,l_sigs,ylcs,xlcs,qls,phis])
    #----------------------------------------------------

    base0 = np.zeros((nnn,nnn,3),'uint8')
    base1 = np.zeros((nnn,nnn,3),'uint8')
    base2 = np.zeros((nnn,nnn,3),'uint8')

    x = 0
    y = 0
    step = 1
    gr_sig = 0.02
    gr_eq = 1.0
    gr_pa = 0.0

    xlc0 = 0.0
    ylc0 = 0.0
    ql0 = 0.7
    rc0 = 0.1
    re0 = 1.0
    phi0 = 0.0
    ap0 = 1.0
    l_sig0 = 0.5
    delta = 1e-8

    LeftButton=0


    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        rotation=pygame.mouse.get_rel()
        buttonpress=pygame.mouse.get_pressed()
        keys = pygame.key.get_pressed()  #checking pressed keys

        #----------------------------------------------------
        if rotation[0] and buttonpress[0] and keys[pygame.K_s]:

            gr_sig=gr_sig-rotation[1]*0.001
            if gr_sig <= 0:
                gr_sig = delta
            if gr_sig >= 1:
                gr_sig = 1.0

        if rotation[0] and buttonpress[0] and keys[pygame.K_p]:
            x += rotation[0]
            y += rotation[1]

        if rotation[0] and buttonpress[0] and keys[pygame.K_e]:
            gr_pa=gr_pa+rotation[0]

            gr_eq=gr_eq+rotation[1]*0.005
            if gr_eq <= 0:
                gr_eq = 0.3
            if gr_eq >= 1:
                gr_eq = 1.0-delta


        #----------------------------------------------------
        if rotation[0] and buttonpress[2] and keys[pygame.K_s]:

            #rc0=rc0+rotation[1]*0.01

            #if rc0 <= 0:
            #    rc0 = delta
            #if rc0 >= 1:
            #    rc0 = 1.0-delta

            #l_sig0 = l_sig0-rotation[1]*0.001
            #if l_sig0 <= 0:
            #    l_sig0 = delta

            re0=re0-rotation[1]*0.01
            if re0 <= 0:
                re0 = delta

            l_sig0 = l_sig0-rotation[1]*0.01
            if l_sig0 <= 0:
                l_sig0 = delta

        if rotation[0] and buttonpress[2] and keys[pygame.K_p]:
            xlc0=xlc0+rotation[0]*0.01
            ylc0=ylc0+rotation[1]*0.01

        if rotation[0] and buttonpress[2] and keys[pygame.K_e]:
            phi0=phi0+rotation[0]

            ql0=ql0+rotation[1]*0.01
            if ql0 <= 0:
                ql0 = 0.3
            if ql0 >= 1:
                ql0 = 1.0-delta


        lpar =  np.asarray([ylc0,xlc0,ql0,rc0,re0,phi0])
        #----------------------------------------------------
        # luminosity parameters for main halo
        glpar  = np.asarray([ap0,l_sig0,ylc0,xlc0,ql0,phi0])
        #----------------------------------------------------

        g_lens = lens_galaxies(xi1,xi2,glpar,glpars)

        base0[:,:,0] = g_lens*256
        base0[:,:,1] = g_lens*128
        base0[:,:,2] = g_lens*0

        if keys[pygame.K_RIGHT]:
            x += step
            if x > 500:
                x = 500
        if keys[pygame.K_LSHIFT] & keys[pygame.K_RIGHT]:
            x += 30*step

        if keys[pygame.K_LEFT]:
            x -= step
            if x < -500:
                x = -500

        if keys[pygame.K_LSHIFT] & keys[pygame.K_LEFT]:
            x -= 30*step

        if keys[pygame.K_UP]:
            y -= step
            if y < -500 :
                y = -500
        if keys[pygame.K_LSHIFT] & keys[pygame.K_UP]:
            y -= 30*step

        if keys[pygame.K_DOWN]:
            y += step
            if y > 500 :
                y = 500
        if keys[pygame.K_LSHIFT] & keys[pygame.K_DOWN]:
            y += 30*step


        #----------------------------------------------
        if keys[pygame.K_MINUS]:
            gr_sig -= 0.01
            if gr_sig <0.01:
                gr_sig = 0.01

        if keys[pygame.K_EQUALS]:
            gr_sig += 0.01
            if gr_sig >0.1:
                gr_sig = 0.1

        #----------------------------------------------
        #parameters of source galaxies.
        #----------------------------------------------
        g_amp = 1.0         # peak brightness value
        g_sig = gr_sig          # Gaussian "sigma" (i.e., size)
        g_ycen = y*2.0/nnn  # y position of center
        g_xcen = x*2.0/nnn  # x position of center
        g_axrat = gr_eq       # minor-to-major axis ratio
        g_pa = gr_pa          # major-axis position angle (degrees) c.c.w. from y axis
        gpar = np.asarray([g_amp, g_sig, g_ycen, g_xcen, g_axrat, g_pa])
        #----------------------------------------------


        g_image,g_lensimage = lensed_images(xi1,xi2,gpar,lpar,lpars)

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

        #font=pygame.font.SysFont(None,30)
        #text = font.render("( "+str(x)+", "+str(-y)+" )", True, (255, 255, 255))
        #screen.blit(text,(10, 10))
        pygame.display.update()
        fpsClock.tick(FPS)


if __name__ == '__main__':
    main()
