#   
# (C) DR0ID 03.2006
# 
# you can contact me at: http://www.mypage.bluewin.ch/DR0ID/pygame
#
# or on IRC: irc.freenode.net 6667 #pygame 
#
#







import pygame
from pygame.locals import *
import lens
from lens import *
import random
import sprite


# try to import psyco, its not needed to run
##psyco=None
##try:
##	import psyco
##	if psyco.__version__>=17105136:
##		from psyco.classes import __metaclass__
##	else:
##		print """Note: Psycho version """, psyco.__version__, """  detected, upgrade to psyco 1.5 for better performance. """
##
##except ImportError:
##	print """Note: No psyco detected. Install psycho to improve performance."""




__author__      = 'DR0ID'
__version__     = '0.1'	# change also top of this file!!
__versionnumber__  = 0,1   # versionnumber seperatly for comparing 
__license__     = 'public domain'
__copyright__   = '(c) DR0ID 03.2006'






#example 
#keys:
# space to attache lens to mouse
# p to attache lens to moving sprite
# c to make lens visible or invisible

def main():
    # setup pygame
    pygame.init()
    screen = pygame.display.set_mode((640,480),1)
    pygame.display.set_caption("toggle lens with the spacebar")
    
    #prepare stuff, load images
    # convert them it will nearly double your fps (frames per second)
    smallI = pygame.image.load("small.png").convert()
    bigI = pygame.image.load("big.png").convert()
    lensImg = pygame.image.load("lens.png").convert_alpha()
    mask = pygame.image.load("lensmask.png").convert()
    # IMPORTANT: set one already one color as transparent with the colorkey
    mask.set_colorkey((255,255,255,255)) # transparent part of the image 
    # lens
    lens = Lens(lensImg,mask, (255, 0, 255), 0 ) #(255,255,255))
    # object which are magnified by the lens
    objlist = []
    for i in range(100):
        objlist.append(MagnifiedObj(lens, smallI, bigI))
        objlist[-1].move(random.random()*640, random.random()*480)
##    mObj = MagnifiedObj(lens,smallI, bigI)
##    mObj.move(100,100) # move them a little bit arround
##    mObj2 = MagnifiedObj(lens,smallI, bigI)
##    mObj2.move(124, 124)
    # group for easier handling and add ouer objects
    # note the order, the lens is on top of all
    g = pygame.sprite.OrderedUpdates()
##    g.add(mObj)
##    g.add(mObj2)
    g.add(objlist)
    g.add(lens)
    lens.changeLayer(1)
    lens.dirty=2

    # a path for demostrating animation and attaching
    path_rect = pygame.Rect(0,0,1,1)
    path_rect.center = (screen.get_width()/2, screen.get_height()/2)
    path_vx = 40
    path_vy = 40
    path_screenW = screen.get_width()
    path_screenH = screen.get_height()
    path_factor = 0.1
    # we need a object with a rect to attache at
    s = pygame.sprite.Sprite
    s.rect = path_rect
           
    # variable
    followPath = False
    
    background = pygame.Surface(screen.get_size()).convert()
    clock = pygame.time.Clock()
    while 1:
        # eventhandling keys
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit() # if you run it from console....
                    return
                elif event.key == K_SPACE:
                    followPath = False
                    if lens.isAttached():
                        pygame.mouse.set_visible(1)
                        lens.dettach()
                    else:
                        pygame.mouse.set_visible(0)
                        lens.attachToObject(pygame.mouse)
                elif event.key == K_p:
                    if followPath:
                        lens.dettach()
                        followPath = False
                    else:
                        lens.attachToObject( s )
                        followPath = True
                elif event.key == K_c:
                    if(lens.isVisible() ):
                        lens.visible(False)
                    else:
                        lens.visible(True)
                elif event.key == K_d:
                    sprite.Renderer.switchDEBUG()
                    
        if path_rect.x > (1-path_factor)*path_screenW or path_rect.x< path_factor*path_screenW:
            path_vx = -path_vx
        if path_rect.y > (1-path_factor)*path_screenH or path_rect.y< path_factor*path_screenH:
            path_vy = -path_vy
        path_rect.move_ip(path_vx, path_vy)
                    
        g.update()
#        g.clear(screen, background)
#        pygame.display.update( g.draw(screen) )
        sprite.Renderer.render()
        pygame.display.set_caption("keys: space, c, p, esc           fps: %4.5s" %(clock.get_fps()) )
#        clock.tick()
        clock.tick()        
    
    
    
#if __name__ == '__main__': main()
if __name__=="__main__":
	main()
