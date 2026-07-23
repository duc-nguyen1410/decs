import numpy as np
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
class Symmetry:
    def __init__(self, s:int=1, sx:int=1, sy:int=1, sz:int=1, ax:float=0, ay:float=0, az:float=0):
        """
        Defines a symmetry operation: S [u(x,y,z)] = s * u(sx*x + ax, sy*y + ay, sz*z + az)

        Parameters:
        s  : Global parity (usually 1 or -1)
        sx : x-reflection (1: no reflection, -1: x -> -x)
        sy : y-reflection (1: no reflection, -1: y -> -y)
        sz : z-reflection (1: no reflection, -1: z -> -z)
        ax : x-shift (translation normalized by Lx)
        ay : y-shift (translation normalized by Ly)
        az : z-shift (translation normalized by Lz)
        """
        self.s = s
        self.sx, self.sy, self.sz = sx, sy, sz
        self.ax, self.ay, self.az = ax, ay, az
    def print(self):
        return f"[{self.s} {self.sx} {self.sy} {self.sz} {self.ax} {self.ay} {self.az}]"

    def load_from_file(self, filename:str):
        with open(filename, 'r') as f:
            logger.info(f"Loading symmetry from file {filename}")
            # Format: s sx sy sz ax ay az
            data = [float(x) for x in f.readline().split()]
            if len(data)==7:
                self.s, self.sx, self.sy, self.sz, self.ax, self.ay, self.az = data
                logger.info("Loaded symmetry: "+self.print())
            else:
                logger.warning(f"File {filename} contains invalid symmetry format! Expected 7 values, got {len(data)}.")
        

    def is_nontrivial(self):
        return not (self.s == 1 and self.sx == 1 and self.sy == 1 and self.sz == 1 and 
                    self.ax == 0 and self.ay == 0 and self.az == 0)
    
    def s(self):
        return self.s
    def set_s(self, s):
        self.s = s 

    def sx(self):
        return self.sx
    def set_sx(self, sx):
        self.sx = sx 
    def sy(self):
        return self.sy
    def set_sy(self, sy):
        self.sy = sy
    def sz(self):
        return self.sz
    def set_sz(self, sz):
        self.sz = sz 
    
    def ax(self):
        return self.ax
    def set_ax(self, ax):
        self.ax = ax 
    def ay(self):
        return self.ay
    def set_ay(self, ay):
        self.ay = ay
    def az(self):
        return self.az
    def set_az(self, az):
        self.az = az 