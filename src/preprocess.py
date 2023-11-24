import numpy as np
from copy import deepcopy
from scipy.ndimage import *

class Preprocess:
    def __init__(self,imgs:list,**kwargs):
        self.imgs           = imgs
        self.min_bright     = kwargs.get("min_bright")
        self.rot_ang        = kwargs.get("rot_ang")

    def rotate_img(self, img:np.ndarray) -> np.ndarray:
        return rotate(img, self.rot_ang)
    
    def normalise_img(self, img:np.ndarray) -> np.ndarray:
        nimg = img / img.max()
        return nimg
    
    def get_binary_img(self, img:np.ndarray) -> np.ndarray:
        bimg  = deepcopy(img)
        bimg[bimg > self.min_bright] = 1
        bimg[bimg <= self.min_bright] = 0
        return bimg
    
    def make_binary_inversion(self, img:np.ndarray) -> np.ndarray:
        bimg = deepcopy(img)
        # Make darker pixels 0
        bimg[bimg <= self.min_bright]= 0
        # Make brighter pixels darker
        bimg[bimg != 0] = 0.25
        # Make darkest pixels brightest
        bimg[bimg == 0] = 1.0
        # Make darker pixels 0
        bimg[bimg == 0.25] = 0
        return bimg

    def blur_img(self, img:np.ndarray, sigma:float=1) -> np.ndarray:
        return gaussian_filter(img,sigma=sigma)
    
    def sobel_filter(self, img:np.ndarray) -> np.ndarray:
        return sobel(img)
    
    def preprocess_img(self, img:np.ndarray) -> np.ndarray:
        #Normalise image using max intensity
        nimg = self.normalise_img(img)
        #Binarise and invert the previously normalised image
        bimg = self.make_binary_inversion(nimg)
        # Rotate image. This would be helpful for further semgent fitting
        rimg = self.rotate_img(bimg)
        #Renormalise image 
        nbimg= self.normalise_img(rimg)
        # Lastly, make final image binary
        fimg = self.get_binary_img(nbimg)
        return fimg
    
    def process_img_stack(self) -> list:
        img_stack = []
        for i in self.imgs:
            img = self.imgs[i]
            #Run image preprocess
            pimg = self.preprocess_img(img)
            img_stack.append(pimg)
        return img_stack