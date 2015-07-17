#
#  oscTuioBullets.py
#  fire bullets using TUIO OSC messages from CCV
#  Bill Thibault May 2009
#  All rights reversed.
#

import sys, pygame
from pygame.locals import *
import numpy
import osc
import os
import platform
if platform.system() == 'Windows':
    os.environ['SDL_VIDEODRIVER'] = 'windib'



pygame.init()

size = width, height = 640,480
speed = [2, 2]

WHITE = (255,255,255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((0,0),FULLSCREEN|DOUBLEBUF)
#screen = pygame.display.set_mode(size,OPENGL|DOUBLEBUF)
#screen = pygame.display.set_mode(size, DOUBLEBUF)

#
#  g l o b a l s
#

X,Y,Z = range(3)
fingers = {}
livingFingers = []

#
#
# O S C
#
#

def tuioCursorHandler(*msg):
    global fingers
    global livingFingers
#    print "tuioHandler", msg
    if msg[0][2] == 'set':
        finger = msg[0][3]
        if not finger in fingers:
            fingers[finger] = [0,0]
# msg[0][4] should be X, but seems to be Y.  why?
        fingers[finger][X] = msg[0][4]
# msg[0]5] should be Y, but seems to be X. why ? and flipped!?!?!?
        fingers[finger][Y] = msg[0][5]
#        print 'set', finger, fingers[finger]
    elif msg[0][2] == 'alive':
        # go through fingerState and mark the living fingers
        livingFingers = msg[0][3:]
#        print 'alive', livingFingers

osc.init()
#osc.listen('192.168.1.7',9001)
osc.listen('127.0.0.1', 3333)
osc.bind ( tuioCursorHandler, "/tuio/2Dcur" )

###########################################
# game objects
#

class Bullet:
    def __init__(self, pos=(0,0), vel=(0,0)):
        x,y = pos
        self.position = numpy.array([float(x),float(y)])
        x,y = vel
        self.velocity = numpy.array([float(x),float(y)])

    def update(self):
        global targets
        global hitTarget
        self.position += self.velocity
        # collide here
        for target in targets:
            d = target.position - self.position
            if d[0]*d[0] + d[1]*d[1] < 64:
                hitTarget = True

    def draw(self):
        pygame.draw.circle(screen,(255,0,0),
                           (int(self.position[0]), int(self.position[1])), 3 )

class Target:
    def __init__(self, pos=(0,0), vel=(0,0)):
        x,y = pos
        self.position = numpy.array([float(x),float(y)])
        x,y = vel
        self.velocity = numpy.array([float(x),float(y)])

    def update(self):
        self.position += self.velocity
        for i in (0,1):
            if self.position[i] < 0:
                self.position[i] += size[i];
            if self.position[i] > size[i]:
                self.position[i] -= size[i];

    def draw(self):
        pygame.draw.circle(screen,(255,255,0),
                           (int(self.position[0]), int(self.position[1])), 8,1 )

class Turret:
    def __init__(self, pos=(0,0)):
        x,y = pos
        self.position = numpy.array([float(x),float(y)])
        self.angle = 0
        #self.surf = pygame.image.load("turret.png").convert()

    def update(self):
        self.angle += 1
        
    def draw(self):
        #rotsurf = pygame.transform.rotate ( self.surf, self.angle )
        #cx = rotsurf.get_width() / 2
        #y = rotsurf.get_height() / 2
        #screen.blit ( rotsurf, tuple(self.position - numpy.array([cx,cy])) )
        pygame.draw.circle ( screen, (0,0,255), self.position, 8, 3 )
        

    
##################################################
#
# M A I N    L O O P         
#
#
clock = pygame.time.Clock()
size = x,y = screen.get_size()
turret = Turret( (x/2,y/2) )
bullets = []
targets = [ Target( (100,0), (0.25, 3.5) ) ]
hitTarget = False

while 1:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT or \
          (event.type == KEYDOWN and event.key == K_ESCAPE): 
            osc.dontListen()
            pygame.quit()
            sys.exit()


    if hitTarget:
        screen.fill((255,0,0))
        hitTarget = False
    else:
        screen.fill(BLACK)

    for finger in livingFingers:
        x = fingers[finger][X] * size[0]
        y = fingers[finger][Y] * size[1]
        pygame.draw.circle(screen, WHITE, (x,y), 5 )
        dir = numpy.array([x,y]) - turret.position
        dir /= 10
        bullets.append ( Bullet ( turret.position, dir ) )

    for bullet in bullets:
        bullet.update()
        bullet.draw()

    for target in targets:
        target.update()
        target.draw()

    turret.update()
    turret.draw()

    pygame.display.flip()


