import numpy as np
from PIL import Image

#im = Image.open("test.jpg")
#pix = np.asarray(im)
#blur = np.full_like(pix,0)
#for(i = 0; i<len(pix))
#    for(j=0; j<len(pix[0]))
#        
#print pix
im = Image.open("test.jpg")
px=im.load()
im2 = im.copy()
px2=im2.load()
sx,sy=im.size
def turnpixel(Nix,Niy):
    for ix in range(sx):
        for iy in range(sy):
            r2=(Nix-ix)**2+(Niy-iy)**2
            if r2<5:
                if sum(px[ix,iy])>100: # rgb sum>100 is considered ON.
                    px2[Nix,Niy]=(255,255,255)                            
                    return
                    # we turned a pixel on, so we are done with it.

for Nix in range(sx):
    for Niy in range(sy):
        px2[Nix,Niy]=(0,0,0)
        turnpixel(Nix,Niy)
im2 = Image.fromarray(px2)
im2.save("blurred.jpg")
