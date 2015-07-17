
import pygame



##class Sprite(pygame.sprite.Sprite):
##    
##    def __init__(self, img, *groups):
##        pygame.sprite.Sprite.__init__(self, *groups)
##        self.image = img
##        self.rect = img.get_rect()
##        self._rendererRemove = Renderer()._sprites.remove # for layer its: Renderer()._sprites[layer].remove
##        self._rendererAppend = Renderer()._sprites.append # Renderer()._sprites[layer].append
##        self.dirty = 0 # has to be last entry because of __setattr__()!!
##        
##    def __setattr__(self, name, value):
##        object.__setattr__(self, name, value)
##        if(name=="dirty" and value>0):
##            self._rendererAppend(self)
##            
##    def kill():
##        try:
##            self._rendererRemove(self)
##        except:
##            pass
##        
    
class NewRect(pygame.Rect):
        
    def __setattr__(self, name, value):
        pygame.Rect.__setattr__(self, name, value)
        if name in ['size','width','height', 'w', 'h']:
            pygame.Rect.__setattr__(self, 'area', self.width * self.height)
        elif 'area' == name:
            raise TypeError, str(self.__class__.__name__) +" area attribute is read only!"
    
    def __getattr__(self, name):
        if 'area' == name:
            pygame.Rect.__setattr__(self, 'area', self.width * self.height)
            return self.area
        else:
            return pygame.Rect.__getattribute__(self, name)
        
    def move(self, x, y):
        return NewRect(pygame.Rect.move(self, x, y))    
        
    def inflate(self, x, y):
        return NewRect(pygame.Rect.inflate(self, x, y))
        
    def inflate_ip(self, x, y):
        pygame.Rect.inflate_ip(self, x, y)
        pygame.Rect.__setattr__(self, 'area', self.width * self.height)
        
    def clamp(self,rect):
        return NewRect(pygame.Rect.clamp(self,rect))
        
    def clamp_ip(self,rect):
        pygame.Rect.clamp_ip(self,rect)
        pygame.Rect.__setattr__(self, 'area', self.width * self.height)
    
    def clip(self,rect):
        return NewRect(pygame.Rect.clip(self,rect))
    
    def clip_ip(self,rect):
        pygame.Rect.clip_ip(self,rect)
        pygame.Rect.__setattr__(self, 'area', self.width * self.height)
    
    def union(self,rect):
        return NewRect(pygame.Rect.union(self,rect))
    
    def union_ip(self,rect):
        pygame.Rect.union_ip(self,rect)
        pygame.Rect.__setattr__(self, 'area', self.width * self.height)
        
    def unionall(self, *rect):
        return NewRect(pygame.Rect.unionall(self,rect))
    
    def unionall_ip(self, *rect):
        pygame.Rect.unionall_ip(self,rect)
        pygame.Rect.__setattr__(self, 'area', self.width * self.height)
    
    def fit(self,rect):
        return NewRect(pygame.Rect.fit(self,rect))
        
    def normalize(self):
        pygame.Rect.normalize(self)
        pygame.Rect.__setattr__(self, 'area', self.width * self.height)
        
    def overlapArea(self,rect):
        """
        Size of the overlaping area.
        """
        return self.clip(rect).area
        
    def unionArea(self,rect):
        """
        Size of the area after union the two rects.
        """
        return self.union(rect).area
        
    def nonOverlapUnionArea(self,rect):
        """
        Size of the non overlaping area of the united rects.
        """
        return self.unionArea(rect)-self.overlapArea(rect)
        
    def nonOverlapArea(self,rect):
        """
        Size of non overlaping area, r1.area+r2.area-r1.clip(r2).area
        """
        return self.area+rect.area-self.clip(rect).area
        
    def percentOverlapOfUnionArea(self,rect):
        return float(self.overlapArea(rect))/self.unionArea(rect)
        
    def percentOverlapOfArea(self,rect):
        return float(self.overlapArea(rect))/(self.area+rect.area)

class NewSprite(pygame.sprite.Sprite):
    
##    def __init__(self, img, *groups):
    def __init__(self, *groups):
        pygame.sprite.Sprite.__init__(self, *groups)
##        self.image = img
##        self.rect = pygame.Rect(-1000,-1000,1,1)#NewRect(img.get_rect())
##        self.area = self.rect.width * self.rect.height # area = widht * height
        self.dirty = 1
##        self._layer = 0 # read only !
        self._renderer = Renderer
        self._renderer.add(self)
            
    def kill(self):
        pygame.sprite.Sprite.kill(self)
        self._renderer.remove(self)
            
    def getLayer(self):
        return self._renderer.getLayerOfSprite(self)
        
    def changeLayer(self, toLayer, removeEmptyLayers = True):
        self._renderer.changeSpriteLayer(self, toLayer, removeEmptyLayers)
        
        
        
        
class _Renderer(object):
        
        
    _instance = 0
        
    def __init__(self):
        # Singleton pattern
        if self._instance is not 0:
            raise ReferenceError , "dont try to instanciate the Renderer, use .Renderer or _Renderer._instance"
        _Renderer._instance = self
        # Sprite lists
        self._sprites = {} # {spr:layerNr, ...} global list of sprites
        self._layers = {} #{layer:[spr,...]}
        # Rect lists
        self._lostRect = [] # removed sprites to clear
        self._spriteOldRect = {} # {sprite:rect}
        # screen, background
        self._screen = 0
        self._background = 0
        self._getScreen = pygame.display.get_surface
        # timing
        self._clock = pygame.time.Clock()
        # init 
        self.render = self._init    # first time it points to _init() after
                                    # that to _render()
        self._initialized = False
        
        
    def _init(self):
        if(pygame.display.get_init()):
            if(not self._initialized):
                self._screen = pygame.display.get_surface()
                if self._screen is None:
                    raise TypeError, "Could not get a valid screen surface: pygame.display.get_surface()!!"
                if 0 == self._background:
                    self._background = pygame.Surface(self._screen.get_size(), self._screen.get_flags(), self._screen)
                    self._background.fill((0,0,0))#(255,0,255)) # TODO: background color
                #self._screen.blit(self._background, (0,0))
                self._lostRect.append(self._screen.get_rect())
                self._initialized = True
                self.render = self._render
                self.render()
                pygame.display.flip()
            else:
                raise UserWarning, "Renderer.init(): you should initialize the renderer only once!"
        else:
            raise ReferenceError , "Renderer.init():you must have pygame.display initialized!!"        
            
    def switchDEBUG(self):
        if(self._initialized):
            if self.render == self._render:
                self.render = self._DEBUGrender
            else:
                self.flip()
                self.render = self._render
        else:
            raise TypeError, "Renderer not initialized!!!"
        
# Basic Algorithme:
# 1. determine dirty rects on screen (optimize dirty rects by overlap checks, tricky)
# 2. blit background area
# 2. find all sprites colliding with dirty rects
# 3. blit all found sprites
        
#dirty == -1 -> -1 do nothing (this ones are not on screen, but still in rendererqueue)
#dirty == 0 -> 0 do nothing (this ones are on screen but dont need repaint)
#dirty == 1 -> 0 clear and draw and reset dirty = 0 (this ones are on screen and need repaint, but perhaps next time not)
#dirty == 2 -> 2 clear and draw and do not reset dirty flag (for animation, allway needs repaint)
###dirty == 3 -> -1 clear but do not draw and reset dirty = -1 (this ones are on screen, but they are erased and not repainted)
        
        
    def _render(self):
        # check if screen has changed
        if self._screen!=self._getScreen():
            # screen has changed, check if new screen is valid
            self._screen = self._getScreen()
            if self._screen is None:
                self._initialized = False
                self._init()
        # screen ok
        
        # speedups
        _blit = self._screen.blit
        _background = self._background 
        _lost = self._lostRect
        _spriteOldRect = self._spriteOldRect
        _update = []
        _update_append = _update.append
        # blit lost rects with background, they are dirty rects on screen
        for r in _lost:
            _update_append( _blit(_background, r, r) )
        _lost[:] = []
        _oldRect = 0
        _unionR = 0
        _clipR = 0
        # find dirty rects on screen and erase dirty sprites from screen(OLD postition!)
        # old and new position of dirty sprite are dirty rects on screen
        for spr in self._sprites.keys():
            _oldRect = _spriteOldRect[spr]
            if 0<spr.dirty and 0!=_oldRect:
##                _update_append( _blit(_background, _oldRect, _oldRect) )
##                _update_append(spr.rect)
                _blit(_background, _oldRect, _oldRect)
                if _oldRect.colliderect(spr):
                    _unionR = _oldRect.union(spr)
                    _clipR = _oldRect.clip(spr)
                    if 0.25<(float(_clipR.w) * _clipR.h)/(float(_unionR.w)*_unionR.h):
                        _update_append(_unionR)
                    else:
                        _update_append(spr.rect)
                        _update_append(_oldRect)
                else:
                    _update_append(spr.rect)
                    _update_append(_oldRect)
##
                if 1 == spr.dirty:
                    spr.dirty = 0
        # find sprites colliding with dirty rects and blit them
        # TODO: optimize here dirty rects on screen( union etc...) or befor update()??
        _layers = self._layers.keys()
        _layers.sort()
        for layer in _layers:#self._layers.keys().sort():
            for spr in self._layers[ layer ]:
                if -1!=spr.rect.collidelist(_update): # intersects with a dirty rect on screen so blit it
                    _spriteOldRect[spr] = self._screen.blit(spr.image, spr.rect)#_blit(spr.image, spr.rect)

        # TODO: optimize here dirty rects on screen( union etc...)
                    
        pygame.display.update(_update)
            
        
            
        
        
    def _DEBUGrender(self):
        print "DEBUGrender"
        # check if screen has changed
        if self._screen!=self._getScreen():
            # screen has changed, check if new screen is valid
            self._screen = self._getScreen()
            if self._screen is None:
                self._initialized = False
                self._init()
        # screen ok
        
        # speedups
        _blit = self._screen.blit
        _background = self._background 
        _lost = self._lostRect
        _spriteOldRect = self._spriteOldRect
        _update = []
        _update_append = _update.append
        # debugging vars
        _debugLostRects = [] # lost rects
        _debugLostRects_append = _debugLostRects.append
        _colorLostR = (255,255,0) # yellow
        _debugOldRects = [] # old position
        _debugOldRects_append = _debugOldRects.append
        _colorOldR = (255,216,66) # orange
        _debugNewRects = [] # new position
        _debugNewRects_append = _debugNewRects.append
        _colorNewR = (0,255,0) # green
        _debugCollidingRects = [] # colliding
        _debugCollidingRects_append = _debugCollidingRects.append
        _colorCollidingR = (0,255,255) # blue
        #_update   # update rects
        _colorUpdateR = (255,0,0) # red
        
        # blit lost rects with background, they are dirty rects on screen
        for r in _lost:
            _update_append( _blit(_background, r, r) )
            _debugLostRects_append(_update[-1])
            print "lost: ", r
        print "update: ", _update, "\n"
        _lost[:] = []
        _oldRect = 0
        # find dirty rects on screen and erase dirty sprites from screen(OLD postition!)
        for spr in self._sprites.keys():
            _oldRect = _spriteOldRect[spr]
            if 0<spr.dirty and 0!=_oldRect:
##                _update_append( _blit(_background, _oldRect, _oldRect) )
##                _update_append(spr.rect)
                _blit(_background, _oldRect, _oldRect)
                if _oldRect.colliderect(spr):
                    _unionR = _oldRect.union(spr)
                    _clipR = _oldRect.clip(spr)
                    if 0.25<(float(_clipR.w) * _clipR.h)/(float(_unionR.w)*_unionR.h):
                        _update_append(_unionR)
                    else:
                        _update_append(spr.rect)
                        _update_append(_oldRect)
                else:
                    _update_append(spr.rect)
                    _update_append(_oldRect)

##
                _debugOldRects_append(_oldRect)
                _debugNewRects_append(spr.rect)
                print "oldRect blit: ", _oldRect
                if 1 == spr.dirty:
                    spr.dirty = 0
        print "update: ", _update, "\n"
##                if spr.dirty==3: # TODO: what for? remove does the same!!
##                    _update_append( _blit(_background, spr.rect) )
##                    spr.dirty = -1
        # find sprites colliding with dirty rects and blit them
        # TODO: optimize here dirty rects on screen( union etc...) or befor update()??
        _layers = self._layers.keys()
        _layers.sort()
        for layer in _layers:#self._layers.keys().sort():
            for spr in self._layers[ layer ]:
                if -1!=spr.rect.collidelist(_update): # intersects with a dirty rect on screen so blit it
                    _spriteOldRect[spr] = self._screen.blit(spr.image, spr.rect)#_blit(spr.image, spr.rect)
                    _debugCollidingRects_append(_spriteOldRect[spr])
                    print "newRect blit: ", spr.rect, spr.__class__.__name__
        print "update: ", _update, "\n"

        # TODO: optimize here dirty rects on screen( union etc...)
                    
        print "update: ", _update, "\n"
        #pygame.display.update(_update)
        
        # debugging code
        for r in _debugLostRects:
            pygame.draw.rect(self._screen,_colorLostR, r, 10) # yellow
        for r in _debugOldRects:
            pygame.draw.rect(self._screen, _colorOldR, r, 8) # orange
        for r in _debugCollidingRects:
            pygame.draw.rect(self._screen, _colorCollidingR, r,6) # blue
        for r in _debugNewRects:
            pygame.draw.rect(self._screen, _colorNewR, r, 4) # green
        for r in _update:
            pygame.draw.rect(self._screen, _colorUpdateR, r,1) # red
        
        pygame.display.flip() # update entire screen (slow!)
        
        _blit(self._background, (0,0)) # rebuild entire screen (slow!)
        for layer in _layers:
            for spr in self._layers[layer]:
                _blit(spr.image, spr)
        self._clock.tick(1) # 1 fps (so one can see what happens on screen)
            
        
    def add(self, *sprites):
        """
        Add a sprite to renderer. One can also pass a iterable sequence of sprites.
        """
        for sprite in sprites:
            if isinstance(sprite, NewSprite):
                self.changeSpriteLayer(sprite, 0)
                self._spriteOldRect[sprite] = 0
            else:
                self.add(*sprite)
            
    def remove(self, sprite):
        """
        Revmoves a sprite from renderer.
        """
        if self._sprites.has_key(sprite):
            self._layers[ self._sprites[sprite] ].remove(sprite)
            del self._sprites[sprite]
            del self._spriteOldRect[sprite]
            self._lostRect.append(sprite.rect)
            
    def changeSpriteLayer(self, sprite, toLayer, removeEmptyLayers=True):
        """
        Changes the layer of a sprite and removes empty layers. Set "removeEmptyLayers" to
        False if you want to retain empty layers.
        """
        if self._sprites.has_key(sprite): # remove sprite from old layer if exists
            self._layers[ self._sprites[sprite] ].remove(sprite)
            if len( self._layers[ self._sprites[sprite] ] )==0 and removeEmptyLayers: # if layer is empty
                del self._layers[ self._sprites[sprite] ]       # remove it
        self._sprites[sprite] = toLayer # assign new layer to sprite
        if self._layers.has_key(toLayer): # and sprite to layer
            self._layers[toLayer].append(sprite) # if layer already exists then append
        else:
            self._layers[toLayer] = [sprite] # else new list
         
    def getLayerOfSprite(self, sprite):
        if self._sprites.has_key(sprite):
            return self._layers[ self._sprites[sprite] ]
        else:
            return 0
        
    def clear(self):
        """
        Removes all sprites from renderer!
        """
        for spr in self._sprites:
            self.remove(spr)
        self.render()
        self._sprites.clear()
        self._layers.clear()
        
    def setBackground(self, bgdSurface):
        """
        Set the background to use to erase the sprites.
        """
        # TODO: stretch or warning if background has not same size as screen???
        self._background = bgdSurface
        
    def setBackgroundColor(self, color):
        """
        Set the color of the background. color = (R,G,B)
        """
        self._background.fill(color)
        
    def flip(self):
        _blit = self._screen.blit
        _blit(self._background, (0,0))
        _layers = self._layers.keys()
        _layers.sort()
        for layer in _layers:
            for spr in self._layers[layer]:
                _blit(spr.image, spr)
        pygame.display.flip()
    
# if you import this module mor than one time only one instance can exist!
if 0==_Renderer._instance:
    Renderer = _Renderer()

    
