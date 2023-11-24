import numpy as np
from scipy.interpolate import interp1d
import ipdb

class Reconstruct:
    def __init__(self,segment_data:dict,labelled_segments,**kwargs):
        self.segment_data      = segment_data
        self.labelled_segments = labelled_segments
        self.polynomial_degree = kwargs.get("polynomial_degree")

    def polynomial_fitting(self,x:np.ndarray,y:np.ndarray,deg:int=3,full:bool=False):
        if self.polynomial_degree:
            deg = self.polynomial_degree
        
        coeffs = np.polyfit(x,y,deg=deg,full=full) #coefficients sorted by decresing power
        return coeffs
    
    def sort_coefficients(self,coeffs:np.ndarray,shift:int=0):
        if shift:
            return np.roll(coeffs[::-1],shift=shift)
        else:
            return coeffs[::-1]
    
    def get_power_sequence(self,n:int):
        return np.arange(0,n,1)
        
    def get_derivative_coeffs(self,coeffs:np.ndarray):
        return np.polyder(coeffs)
    
    def get_new_segment_lenght(self,start:float,end:float,points:int):
        return np.linspace(start,end,points)
        
        
    def extract_segment(self,t:int) -> tuple:
        segment_id = self.labelled_segments[t]
        pos = self.segment_data[t][segment_id].get("pos")
        return (pos[:,0], pos[:,1], pos[:,-1])
    
    def fit_segment(self,t:int,points:int=12) -> tuple:
        x, y, ids = self.extract_segment(t=t)
        
        f = interp1d(x,y,kind="nearest")
        x_ = np.linspace(x.min(),x.max(),points).reshape(-1,1)
        fit = np.concatenate((x_,f(x_)),axis=1)
        return fit
        
    def get_arc_lenght(self,t:int) -> tuple:
        fit = self.fit_segment(t)
        
        deltas = np.gradient(fit,axis=0)
        lenghts = np.linalg.norm(deltas,axis=1)
        arc_lenght = np.sum(lenghts, axis=0)
        
        return (arc_lenght,fit)
    
    def get_curvature(self,fit):
        prime = np.gradient(fit,axis=0)
        double_prime = np.gradient(prime,axis=0)
        
        dx,dy = prime[:,0], prime[:,1]
        ddx,ddy = double_prime[:,0], double_prime[:,1]
        
        curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**(3/2)
        return curvature
    
    def process_all_segments(self):
        reconst = {}
        for t in self.segment_data:
            reconst.setdefault(t,{})
            
            arc_lenght, fit = self.get_arc_lenght(t)
            curvature = self.get_curvature(fit)

            reconst[t]["arc_lenght"] = arc_lenght
            reconst[t]["curvature"] = curvature
            reconst[t]["fit"] = fit
            
        return reconst