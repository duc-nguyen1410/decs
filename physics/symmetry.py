import numpy as np
class Symmetry:
    def __init__(self, s=1, sx=1, sy=1, sz=1, st=1, ax=0, ay=0, az=0):
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

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            # Format: s sx sy sz ax ay az
            data = [float(x) for x in f.readline().split()]
        return cls(s=data[0], sx=data[1], sy=data[2], sz=data[3], 
                   ax=data[4], ay=data[5], az=data[6])

    def is_nontrivial(self):
        return not (self.s == 1 and self.sx == 1 and self.sy == 1 and self.sz == 1 and 
                    self.ax == 0 and self.ay == 0 and self.az == 0)