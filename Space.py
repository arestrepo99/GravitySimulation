from TensorKernel import Tensor, Kernel, queue, program
import numpy as np

class Space():
    def __init__(self, pos, vel, G = 6.7e-11):
        #Constants
        self.N = pos.shape[0]
        self.G = G
        
        #GPU Buffers
        self.pos = Tensor(pos)
        self.vel = Tensor(vel)
        self.imageShape = (np.int32(900),np.int32(1200))
        self.image = Tensor(np.zeros(self.imageShape),dtype=np.uint8)
        self.index = Tensor(pos.shape, dtype= np.uint32)

        #GPU Kernels
        self.applyStep = Kernel(program.applyStep,
                (self.N,), (self.pos,self.vel,self.N,self.G))
        self.getIndices = Kernel(program.getIndices, 
                (self.N,),
                 (self.pos,self.image, self.index,self.N, *self.imageShape))
        self.computeImage = Kernel(program.computeImage,
                self.image.shape, 
                 (self.pos,self.image,self.index, self.N, *self.imageShape))
        self.frame = 0

    def step(self, dt, zoom = 1.0):
        self.applyStep(dt)
        self.getIndices(float(zoom))
        self.computeImage()
        self.frame += 1
        return self.image.get()