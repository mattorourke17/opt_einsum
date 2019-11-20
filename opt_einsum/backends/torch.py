"""
Required functions for optimized contractions of numpy arrays using pytorch.
"""

import numpy as np
import os

from ..parser import convert_to_valid_einsum_chars
from ..sharing import to_backend_cache_wrap
from ..torch_ad_helper.tensordot import Tensordot as TDOT

__all__ = ["transpose", "tensordot", "to_torch", "build_expression", "evaluate_constants"]

_TORCH_DEVICE = None
_TORCH_HAS_TENSORDOT = None

#save tensordot intermediates to disk ('True') or try to store them all in RAM ('False')
TDOT_TO_DISK = os.environ.get('TORCH_TENSORDOT_TO_DISK', 'False')

_torch_symbols_base = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _get_torch_and_device():
    global _TORCH_DEVICE
    global _TORCH_HAS_TENSORDOT

    if _TORCH_DEVICE is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _TORCH_DEVICE = torch, device
        _TORCH_HAS_TENSORDOT = hasattr(torch, 'tensordot')

    return _TORCH_DEVICE

# wrap the custom tensordot class
def diff_tdot(A, B, axes):
    if TDOT_TO_DISK in ['True', 'TRUE', 'true']:
        return TDOT.apply(A, B, axes, True)
    elif TDOT_TO_DISK in ['False', 'FALSE', 'false']:
        return TDOT.apply(A, B, axes, False)
    else:
        raise ValueError("""invalid specification for TORCH_TENSORDOT_TO_DISK.
                        Use strings 'True' or 'False'.""")


def transpose(a, axes):
    """Normal torch transpose is only valid for 2D matrices.
    """
    return a.permute(*axes)



def tensordot(x, y, axes=2):
    """Simple translation of tensordot syntax to einsum.
    """
    return diff_tdot(x, y, axes)


@to_backend_cache_wrap
def to_torch(array):
    torch, device = _get_torch_and_device()

    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device)

    return array


def build_expression(_, expr):  # pragma: no cover
    """Build a torch function based on ``arrays`` and ``expr``.
    """

    def torch_contract(*arrays):
        torch_arrays = [to_torch(x) for x in arrays]
        torch_out = expr._contract(torch_arrays, backend='torch')

        if torch_out.device.type == 'cpu':
            return torch_out.numpy()

        return torch_out.cpu().numpy()

    return torch_contract


def evaluate_constants(const_arrays, expr):
    """Convert constant arguments to torch, and perform any possible constant
    contractions.
    """
    const_arrays = [to_torch(x) for x in const_arrays]
    return expr(*const_arrays, backend='torch', evaluate_constants=True)
