# Author: Matt O'Rourke
# Custom differentiable tensordot implementation to allow for disk storage of
# intermediates. This code relies heavily on the concepts from the HIPS/autograd
# implementation (github.com/HIPS/autograd).


import numpy as np
import itertools
import string
import torch
import os, sys
storedir = '/home/matt/labcode/quimb_testing'

RAND_UUIDS = map("".join, itertools.product(string.hexdigits, repeat=7))

def rand_uuid():
    return next(RAND_UUIDS)

class Tensordot(torch.autograd.Function):
    @staticmethod
    def forward(self, A, B, axes, to_disk):
        C = torch.tensordot(A, B, dims=axes)
        self.to_disk = to_disk
        self.axes = axes

        if not self.to_disk:
            self.save_for_backward(A, B)
        else:
            st = rand_uuid()
            torch.save(A, os.path.join(storedir, f'store/tdot/A{st}.pt'))
            torch.save(B, os.path.join(storedir, f'store/tdot/B{st}.pt'))
            self.identify = st

        return C

    @staticmethod
    def backward(self, dC):
        if not self.to_disk:
            A, B = self.saved_tensors
        else:
            st = self.identify
            A = torch.load(os.path.join(storedir, f'store/tdot/A{st}.pt'))
            B = torch.load(os.path.join(storedir, f'store/tdot/B{st}.pt'))

        axes = self.axes
        A_ndim = len(A.size())
        B_ndim = len(B.size())

        # first do the adjoint of A
        if B_ndim == 0:
            dA = dC * B

        else:
            dC_axes = list(np.arange(len(dC.size())))
            if type(axes) is int:
                axes = max(axes, 0)
                B_axes = list(np.arange(B_ndim))
                dA = torch.tensordot(dC, B, dims=[dC_axes[A_ndim-axes:], B_axes[axes:]])
            else:
                axes0 = [axes[0]] if type(axes[0]) is int else axes[0]
                axes1 = [axes[1]] if type(axes[1]) is int else axes[1]
                axes = [axes0, axes1]
                A_axes = np.arange(A_ndim)
                B_axes = np.arange(B_ndim)
                summed_axes = [np.asarray(axes[0], dtype='int64') % A_ndim,
                               np.asarray(axes[1], dtype='int64') % B_ndim]
                other_axes  = [np.delete(A_axes, summed_axes[0]),
                               np.delete(B_axes, summed_axes[1])]
                perm = np.argsort(np.concatenate(
                        (other_axes[0], summed_axes[0][np.argsort(summed_axes[1])])))
                out = torch.tensordot(dC, B, dims=[dC_axes[len(other_axes[0]):],
                                                                list(other_axes[1])])
                dA = out.permute(list(perm))

        # now do the adjoint of B
        if A_ndim == 0:
            dB = dC * A

        else:
            dC_axes = list(np.arange(len(dC.size())))
            if type(axes) is int:
                axes = max(axes, 0)
                A_axes = list(np.arange(A_ndim))
                dB = torch.tensordot(A, dC, dims=[A_axes[:A_ndim-axes],
                                                        dC_axes[:A_ndim-axes]])
            else:
                axes0 = [axes[0]] if type(axes[0]) is int else axes[0]
                axes1 = [axes[1]] if type(axes[1]) is int else axes[1]
                axes = [axes0, axes1]
                A_axes = np.arange(A_ndim)
                B_axes = np.arange(B_ndim)
                summed_axes = [np.asarray(axes[0], dtype='int64') % A_ndim,
                               np.asarray(axes[1], dtype='int64') % B_ndim]
                other_axes  = [np.delete(A_axes, summed_axes[0]),
                               np.delete(B_axes, summed_axes[1])]
                perm = np.argsort(np.concatenate(
                    (summed_axes[1][np.argsort(summed_axes[0])], other_axes[1])))
                out = torch.tensordot(A, dC, dims=[list(other_axes[0]),
                                                    dC_axes[:len(other_axes[0])]])
                dB = out.permute(list(perm))

        return dA, dB, None, None


def test_tdot():
    L, M, N = 5, 40, 10
    torch.manual_seed(2)
    t1 = torch.rand(M, N, L, dtype=torch.float64, requires_grad=True)
    t2 = torch.rand(M, L, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(Tensordot.apply, (t1, t2, [[0,1],[0,2]], True), eps=1e-6, atol=1e-4))
    print("Test Pass!")

if __name__=='__main__':
    test_tdot()
