import pyopencl as cl
import numpy as np

context = cl.Context()
queue = cl.CommandQueue(context)
program = cl.Program(context, open('kernels.cl').read()).build()

class Tensor:
    def __init__(self, input, shape = None, dtype = np.float32 ):
        self.dtype = dtype
        if isinstance(input, cl.Buffer):
            self.buffer = input
            self.shape = shape
        elif isinstance(input, np.ndarray):
            self.shape = input.shape
            self.buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE,
                        size=np.prod(self.shape)*dtype(1).nbytes)
            self.set(input)
        elif isinstance(input, tuple):
            self.shape = input
            self.buffer = cl.Buffer(context,cl.mem_flags.READ_WRITE,
                        size=np.prod(self.shape)*dtype(1).nbytes)
        else:
            raise(TypeError("Expected buffer, numpy array, or shape"))
    
    def get(self):
        out = np.empty(self.shape).astype(self.dtype)
        cl.enqueue_copy(queue, src=self.buffer, dest=out)
        return out

    def __repr__(self) -> str:
        return self.get().__str__()

    def set(self,values):
        cl.enqueue_copy(queue, src=values.flatten().astype(self.dtype), 
                    dest=self.buffer)


class Kernel():
    def __init__(self, function, globalSize, staticParams):
        self.function = function
        self.staticParams  = staticParams
        self.globalSize = globalSize
        self.localSize = None
        self.chunks = 1
        self.globalOffset = None
        
    def __call__(self, *params):
        def getBuffers(params):
            out = []
            for param in params:
                if isinstance(param,Tensor):
                    out.append(param.buffer)
                elif isinstance(param,int):
                    out.append(np.int32(param))
                elif isinstance(param,float):
                    out.append(np.float32(param))
                else:
                    out.append(param)
            return tuple(out)
        if self.localSize is None:
            self.optimize(params)
        self.function(queue, self.globalSize, self.localSize, 
            *getBuffers(params), *getBuffers(self.staticParams))

    def optimize(self, params, reps = 2, debug = True):        
        self.localSize = tuple([1]*len(self.globalSize))

