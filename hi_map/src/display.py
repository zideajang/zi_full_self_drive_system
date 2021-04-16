import cv2
import sdl2
import sdl2.ext


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
   