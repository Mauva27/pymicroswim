import numpy as np
import cv2
import os
import pickle
from PIL import Image

def load_images(imgs:str,crop:bool,cut_npx:int):
    all_imgs = Image.open(imgs)
    img_list = []

    for i in range(all_imgs.n_frames):
        all_imgs.seek(i)
        img_list.append(np.array(all_imgs.convert('L')))
    img_array = np.array(img_list)

    img_stack = {}
    for j in range(img_array.shape[0]):
        this_image = img_array[j]
        if crop:
            this_image_ = this_image[cut_npx : this_image.shape[0] - cut_npx, cut_npx : this_image.shape[1] - cut_npx]
        else:
            this_image_ = this_image
        img_stack[j] = this_image_
    return img_stack

def load_movie_frames(directory,filename,frame,fmt):
    """
    returns the frame in movie as array
    """
    vidcap = cv2.VideoCapture(directory+filename+fmt)
    success, count = 1, 0
    while success:
        success, image = vidcap.read()
        if (count == frame) and success:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        count += 1
    return None

def load_piezo_actuation(filename:str) -> tuple:
    with open(filename, "r") as f:
        lines = f.read().split('\n')

        time = 0
        positions = [] 

        for line in lines:
            if "DEL" in line:
                # Read time in ms and convert to sec
                t = float(line.replace("DEL ","")) * 1e-3
                time = t
            elif "MOV" in line:
                # Read the z-position
                p = float(line.replace("MOV 3 ",""))
                positions.append(p)
        return (time, positions)
    
def complete_z_positions(duration:float,positions,nimgs:int) -> tuple:
    ncycles = nimgs // len(positions)

    zpos = []
    for n in range(ncycles):
        zpos.append(positions)

    zpos = np.hstack(zpos)

    if len(zpos) < nimgs:
        missing = nimgs - len(zpos)
        zpos = np.concatenate((zpos,positions[:missing]))
    timesteps = np.linspace(0,len(zpos) * duration, len(zpos))
    return (zpos,timesteps)



def open_pickle(filename:str,encoding:str=None):
    '''
    Funtion to read pickle files

    Parameters
    ----------
    filename    : str. The filename extension of the file including the ".p" format.
    endcoding   : bool (optional). Uses "latin1" encoding to readl all pickle files. None default. 

    Returns
    ----------
    p           : pickle. Data read from filename

    '''

    with open(filename, 'rb') as f:
        u = pickle.Unpickler(f)
        if encoding:
            u.encoding = 'latin1'
        else: 
            pass
        p = u.load()
        return p



