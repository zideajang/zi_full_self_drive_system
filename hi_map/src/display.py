import cv2
import sdl2
import sdl2.ext

import pygame
from pygame.locals import *

import numpy as np

class Display2(object):

    def __init__(self,W,H):
        self.W = W
        self.H = H

        self.FPS = 60
        self.screen = pygame.display.set_mode((self.W,self.H))

    def show(self,frame):
        self.screen.fill(0)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame,(0,0))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == KEYDOWN:
                pygame.quit()
                cv2.destroyAllWindows()
        
              

        

class Display(object):

    def __init__(self,W,H):
        
        sdl2.ext.init()

        self.W = W
        self.H = H

        self.window = sdl2.ext.Window("SLAM",size=(self.W,self.H),position=(-500,-500))
        self.window.show()
        
    def show(self,frame):
        
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        # pixels3d
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        # 
        surf[:,:,0:3] = frame.swapaxes(0,1)
        self.window.refresh()

if __name__ == "__main__":
    pass
   