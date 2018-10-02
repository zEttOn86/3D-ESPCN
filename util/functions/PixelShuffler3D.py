from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

class PixelShuffler3D(function.Function):
    """3D Pixel Shuffler"""
    def __init__(self, r):
        self.r = r

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)#argument check
        type_check.expect(in_types[0].dtype == numpy.float32,
                          in_types[0].ndim == 5
                          )
        
    def forward(self, inputs):
        self.retain_inputs(())
        X, = inputs#X = inputs[0]
        xp = cuda.get_array_module(X)
        bsize, c, d, e, f = X.shape
        c //= self.r**3
        X = xp.transpose(X,(0,2,3,4,1))
        X = xp.reshape(X,(bsize, d, e, f, self.r, self.r, self.r, c))
        X = xp.transpose(X, (0, 1, 4, 2, 5, 3, 6, 7))
        X = xp.reshape(X,(bsize, d*self.r, e*self.r, f*self.r, c))
        X = xp.transpose(X, (0, 4, 1, 2, 3))
        return X,

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs#up layer delta
        xp = cuda.get_array_module(gy)
        bsize, c, d, e, f = gy.shape
        gy = xp.transpose(gy, (0,2,3,4,1))
        gy = xp.reshape(gy, (bsize, d//self.r, self.r, e//self.r, self.r, f//self.r, self.r, c))
        gy = xp.transpose(gy, (0,1,3,5,2,4,6,7))
        gy = xp.reshape(gy, (bsize, d//self.r, e//self.r, f//self.r, (self.r**3) * c))
        gy = xp.transpose(gy, (0, 4, 1, 2, 3))
        return gy,

def pixelshuffler3d(X, r):
    """Computes the 3D Pixel Shuffler
    Update:(20170928) bug fix
    Args:
        X (:class:`~chainer.Variable` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
            Variable holding a 5d array of shape
            ``(batch, channel * r * r * r, dim1, dim2, dim3)``.
        r (int): the upscaling factor.

    Returns:
        ~chainer.Variable:
            A variable holding the upscaled array from
            interspersed depth layers. The shape is
            ``(batch, channel, dim1 * r, dim2 * r, dim3 * r)``.
    ...Example
    >>> X = np. arange(216).reshape(1, 8, 3, 3, 3).astype("f")
    >>>X.shape
    (1,8,3,3,3)
    >>>Y = module1.pixelshuffler3d(X, 2)
    >>>Y.shape
    (1,1,6,6,6)
    """
    return PixelShuffler3D(r)(X)