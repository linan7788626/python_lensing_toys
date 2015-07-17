#   
# (C) DR0ID 03.2006
# 
# you can contact me at: http://www.mypage.bluewin.ch/DR0ID/pygame
#
# or on IRC: irc.freenode.net 6667 #pygame 
#
#

""" 
    This module will provide a faked magnifying lens effect.
    
"""



import pygame
from sprite import NewSprite as Sprite
#from pygame.locals import *



__author__      = 'DR0ID'
__version__     = '0.1'	# change also top of this file!!
__versionnumber__  = 0,1   # versionnumber seperatly for comparing 
__license__     = 'public domain'
__copyright__   = '(c) DR0ID 03.2006'




class Lens(Sprite):
    """
        Special "sprite", need different "images" for representation.
        
        lensimg is a transparent image representing the glass.
        
        mask is for clipping shape of the glass. It needs to bee made with two
        colors and already set one as colorkey(transparent area, where the glass
        goes). The other color is the maskcolorkey.
        
        maskcolorkey it to clip the mask.
        
        background is the background color or image(not magnified).
        I recomend to use one MagnifiedObj sprite for background.
    """
        
    def __init__(self, lensimg, mask, maskcolorkey, background=0, groups=()):
        """
            lensimg: image with alpha representing glas
            mask: two colored shape of glass with already one colorkey enabled
            maskcolorkey: the other color of the mask
            background: color or image when lens does not hit a MagnifiedObj sprite
                (I recomend to use one MagnifiedObj sprite for the background too)
        """
        Sprite.__init__(self, groups)
        self.objects = pygame.sprite.Group()
        self.lupeimg = lensimg.convert_alpha()
        if type(background)==type(pygame.Surface):
            self.bgd = background
        else:
            self.bgd = pygame.Surface(lensimg.get_size())
            self.bgd.fill(background)
            self.bgd = self.bgd.convert()
        self.image = pygame.Surface(lensimg.get_size()).convert()
        self.rect = self.lupeimg.get_rect()
        self.mask = mask
        self.mask.set_alpha(None)
        self.maskcolorkey = maskcolorkey
        self.on = False
        self.attachedTo = None
        self.visible(False)
        
    def update(self):
        """
            There the actually bliting is done. (attention to the alphas and colorkeys!)
        """
        if self.on:
            if self.attachedTo==pygame.mouse :
                self.moveCenterTo( *pygame.mouse.get_pos() )
            else:
                if self.attachedTo is not None:
                    self.moveCenterTo( *self.attachedTo.rect.center )
            coll = pygame.sprite.spritecollide(self, self.objects, False) # get colliding object to magnifie
            self.image.blit(self.bgd, (0,0))
            self.image.set_colorkey(None)
            self.image.set_alpha(None)
            if len(coll):
                for c in coll:
                    sourceRect = pygame.Rect(self.rect)
                    sourceRect.centerx = int( round( (sourceRect.centerx-c.rect.x)*c.xScale) )
                    sourceRect.centery = int( round( (sourceRect.centery-c.rect.y)*c.yScale) )
                    dx = 0
                    dy = 0
                    if sourceRect.x<0:
                        dx = -sourceRect.x
                    if sourceRect.y<0:
                        dy = -sourceRect.y
                    sourceRect = sourceRect.clip(c.bigRect)
                    self.image.blit(c.bigImg, (dx, dy), sourceRect)
            else:
                self.image.blit( self.bgd, (0,0) ) # if no object collides show background
            self.image.blit( self.mask, (0,0) )
            self.image.set_colorkey(self.maskcolorkey) 
            self.image.blit( self.lupeimg, (0,0) )
            
    def move(self, dx, dy):
        """
            moves (nudge) the sprite about dx, dy
        """
        self.rect.move_ip(dx, dy)
        
    def moveCenterTo(self, x, y):
        """
            moves the center of the rect to x, y
        """
        self.rect.centerx = x
        self.rect.centery = y
        
    def moveTo(self, x, y):
        """
            moves the topleft of rect to x, y
        """
        self.rect.x = x
        self.rect.y = y
        
    def register(self, obj):
        """
            hook method for the MagnifiedObj sprites
        """
        self.objects.add(obj)
        
    def attachToObject(self, obj):
        """
            with this method one can attache the lens to a nother sprite
        """
        self.on = True
        self.attachedTo = obj
        
    def dettach(self):
        """
            detach from former atteched sprite
        """
        obj = self.attachedTo
        self.attachedTo = None
        return obj
    
    def isAttached(self):
        """
            return true if it is attached to a sprite, otherwise false
        """
        return self.attachedTo != None
    
    def visible(self, onScreen):
        """
            visible(1) enables lens
            visible(0) clear lens
            set if lens is visible on screen (actually a fully transparent surface is used)
        """
        if onScreen:
            self.on = True
        else:
            self.on = False
            self.image = pygame.Surface(self.rect.size)
            self.image.set_alpha(0)
            self.image = self.image.convert_alpha()
        
    def isVisible(self):
        """
            
        """
        return (self.on==True)
        
    def info(self):
        """
            prints some info to console
        """
        print "Lens"
        print "rect ", self.rect
        print "onscreen ",self.on
        print "objects ", self.objects
        print "attached ", self.isAttached(), self.attachedTo


class MagnifiedObj(Sprite):
    """
        These are the objects which are magnified with the lens sprite.
        Because it is a faked effect, one need two identical images different
        in size.
    """
    
    def __init__(self, lens, smallImg, bigImg, groups = ()):
        """
            lens: is a sprite of type Lens ( no type check is done, but only works with Lens)
            smallImg: the small image, actually drawn on screen
            bigImg: the big image used by the lens to fake the magnification
            groups: groups to add these sprite (like pygame.sprite.Sprite)
        """
        Sprite.__init__(self, groups)
        self.image = smallImg
        self.rect = smallImg.get_rect()
        self.bigImg = bigImg
        self.bigRect = bigImg.get_rect()
        self.xScale = bigImg.get_width()/float(smallImg.get_width())
        self.yScale = bigImg.get_height()/float(smallImg.get_height())
        lens.register(self)
        
    def move(self, dx, dy):
        """
            move (nudge) about dx, dy
        """
        self.rect.move_ip(dx, dy)
        
    def moveTo(self, x, y):
        """
            move topleft corner to x, y
        """
        self.rect.x = x
        self.rect.y = y
         
    def info(self):
        """
            print some info about these sprite to console
        """
        print "MagnifiedObj"
        print "rect ",self.rect
        print "bigrect ", self.bigRect
        print "xScale ", self.xScale
        print "yScale ", self.yScale
        
    


