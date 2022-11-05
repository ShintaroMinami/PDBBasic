import numpy as np
import torch
from einops import rearrange, repeat
from .utils import unsqueeze


def distance_matrix(coord1, coord2=None, no_sqrt=False, eps=1e-10):
    coord1 = unsqueeze(coord1, 0) if len(coord1.shape) < 3 else coord1
    if coord2 is None:
        coord2 = coord1
    else:
        coord2 = unsqueeze(coord2, 0) if len(coord2.shape) < 3 else coord2
    dmat = _calc_distmat2(coord1, coord2)
    if torch.is_tensor(coord1)==torch.is_tensor(coord2)==True:
        return dmat if no_sqrt==True else torch.sqrt(dmat+eps)
    else:
        return dmat if no_sqrt==True else np.sqrt(dmat)


def torsion_angles(coord):
    coord = unsqueeze(coord, 0) if len(coord.shape) < 4 else coord
    shape = coord.shape
    flat_ncac = rearrange(coord[:,:,0:3,:], 'b l a c -> (b l a) c')
    if torch.is_tensor(coord):
        dihedral = _points2dihedral_torch(flat_ncac[0:-3], flat_ncac[1:-2], flat_ncac[2:-1], flat_ncac[3:])
    else:
        dihedral = _points2dihedral_numpy(flat_ncac[0:-3], flat_ncac[1:-2], flat_ncac[2:-1], flat_ncac[3:])
    dihedral = rearrange(dihedral, '(b l d) -> b l d', b=shape[0], l=shape[1], d=3)
    dihedral[:,0,0] = 0.0
    dihedral[:,-1,1:] = 0.0
    return dihedral.squeeze()


def _calc_distmat2(co1, co2):
    mat1 = repeat(co1, 'b x c -> b x y c', y=co2.shape[1])
    mat2 = repeat(co2, 'b y c -> b x y c', x=co1.shape[1])
    if torch.is_tensor(co1)==torch.is_tensor(co2)==True:    
        return ((mat1 - mat2)**2).sum(dim=3)
    else:
        return ((np.array(mat1) - np.array(mat2))**2).sum(axis=3)


def _points2dihedral_torch(c1, c2, c3, c4, eps=0.00001):
    u1, u2, u3 = c2 - c1, c3 - c2, c4 - c3
    dihedral = torch.atan2(
        ((torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1)).sum(dim=-1),
        (torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1)).sum(dim=-1)
    )
    dihedral = torch.min(dihedral, (np.pi-eps)*torch.ones_like(dihedral))
    dihedral = torch.max(dihedral, -(np.pi-eps)*torch.ones_like(dihedral))
    dihedral = torch.nn.functional.pad(dihedral, (1,2), 'constant', 0)    
    return dihedral


def _points2dihedral_numpy(c1, c2, c3, c4, eps=0.0001):
    u1, u2, u3 = c2 - c1, c3 - c2, c4 - c3
    dihedral = np.arctan2(
        ((np.linalg.norm(u2, axis=-1, keepdims=True) * u1) * np.cross(u2,u3, axis=-1)).sum(-1),
        (np.cross(u1,u2, axis=-1) * np.cross(u2, u3, axis=-1)).sum(-1)
    )
    dihedral = np.minimum(dihedral, (np.pi-eps)*np.ones_like(dihedral))
    dihedral = np.maximum(dihedral, -(np.pi-eps)*np.ones_like(dihedral))
    dihedral = np.pad(dihedral, (1,2), mode='constant')
    return dihedral

