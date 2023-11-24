import numpy as np
from sklearn.cluster import DBSCAN

class Segment:
    def __init__(self, imgs:list,zpositions:np.ndarray,timesteps:np.ndarray,**kwargs):
        self.imgs           = imgs
        self.zpos           = zpositions
        self.timesteps      = timesteps
        self.min_bright     = kwargs.get("min_bright")
        self.neigh_distance = kwargs.get("neigh_dist")
        self.min_neighs     = kwargs.get("min_neighs")

    def normalise_img(self, img:np.ndarray):
        nimg = img / img.max()
        return nimg

    def find_segment_pos(self, img:np.ndarray, zpos:float=0, timestep:float=0, min_bright:float=0.45):
        if self.min_bright:
            roi = img >= self.min_bright
        else:
            roi = img >= min_bright
        r = np.vstack(np.where(roi)).T
        z = np.linspace(zpos,zpos,len(r)).reshape(-1,1)
        t = np.linspace(timestep,timestep,len(r)).reshape(-1,1)
        ids = np.arange(len(r)).reshape(-1,1)


        pos = np.zeros_like(r)
        #swap coordinates to match with source images
        pos[:,0] = r[:,1]
        pos[:,1] = r[:,0]

        ext_pos = np.concatenate((pos,z,t,ids),axis=1)
        return ext_pos

    def get_segments(self, pos:np.ndarray, neigh_distance:float=1, min_neighs:int=1):
        if self.neigh_distance and self.min_neigh:
            s = DBSCAN(eps = self.neigh_distance, min_samples = self.min_neighs).fit(pos[:,:2])
        else:
            s = DBSCAN(eps = neigh_distance, min_samples = min_neighs).fit(pos[:,:2])

        segments = np.unique(s.labels_[s.labels_ >= 0])

        sdict = {}
        for segment in segments:
            this = s.labels_ == segment
            sdict[segment] = {}

            com = np.array([np.mean(pos[this][:,0]), np.mean(pos[this][:,1])])
            sdict[segment]['com'] = com
            sdict[segment]['pos'] = pos[this]
        return sdict

    def run_segmentation(self, img:np.ndarray, zpos:float, timestep:float):
        nimg = self.normalise_img(img)
        pos  = self.find_segment_pos(img=nimg,zpos=zpos,timestep=timestep)
        segments = self.get_segments(pos)
        return segments
    
    def run_segmentation_all(self):
        segments = {}
        for i,img in enumerate(self.imgs):
            seg = self.run_segmentation(img=img,zpos=self.zpos[i],timestep=self.timesteps[i])
            segments[i] = seg
        return segments

    def find_this_segment(self, sdict:dict,segment_size:int):
        for segment in sdict:
            pos = sdict[segment].get("pos")

            if len(pos) >= segment_size:
                return segment
            else:
                pass
            
    def find_all_segments_time(self,sdict:dict,segment_size:int):
        soi = {} #segments of interest
        for i in sdict:
            segment_id = self.find_this_segment(sdict[i],segment_size)
            if segment_id != None:
                soi[i] = segment_id
            else:
                print(f"Segment not found in image {i}")
        return soi
                

