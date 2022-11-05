import numpy as np
import torch
from einops import rearrange, repeat
from itertools import product
from pytorch3d.transforms import random_quaternions, quaternion_to_matrix

ANGLE_N_CA_C = torch.tensor(np.deg2rad(111.2, dtype=np.float32))
ANGLE_CA_C_C = torch.tensor(np.deg2rad(116.2, dtype=np.float32))
ANGLE_C_N_CA = torch.tensor(np.deg2rad(124.2, dtype=np.float32))
ANGLE_CA_C_O = torch.tensor(np.deg2rad(120.8, dtype=np.float32))
TORSION_N = torch.tensor(np.deg2rad(120, dtype=np.float32))
TORSION_CA = torch.tensor(np.deg2rad(180, dtype=np.float32))
TORSION_C = torch.tensor(np.deg2rad(180, dtype=np.float32))
TORSION_PI = torch.tensor(np.deg2rad(180, dtype=np.float32))
LENGTH_N_CA = 1.458
LENGTH_CA_C = 1.523
LENGTH_C_N  = 1.329
LENGTH_C_O  = 1.231

UNIT_NCAC = np.array([
        [-0.5229,  1.3598,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [ 1.5244,  0.0000,  0.0000]
    ], dtype=np.float32)

UNIT_CACN = np.array([
        [-0.6839,  1.3617,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [-0.6656, -1.0349,  0.0000],
        [ 1.3288,  0.0000,  0.0000]
    ], dtype=np.float32)


def coord2rotation(coord, eps=1e-8):
    x1 = coord[:,2] - coord[:,1]
    x1 = x1 / (repeat(torch.linalg.norm(x1, dim=1), 'l -> l c', c=3)+eps)
    x2 = coord[:,0] - coord[:,1]
    x2 = x2 - torch.einsum('l c, l -> l c', x1, (torch.einsum('l c, l c -> l', x1, x2)))
    x2 = x2 / (repeat(torch.linalg.norm(x2, dim=1), 'l -> l c', c=3)+eps)
    x3 = torch.cross(x1, x2)
    x3 = x3 / (repeat(torch.linalg.norm(x3, dim=1), 'l -> l c', c=3)+eps)
    return torch.stack([x1, x2, x3], dim=1).transpose(1,2)


def coord2translation(coord):
    return coord[:,1]


def coord_to_frame(coord: torch.Tensor) -> tuple:
    org_type = type(coord)
    coord = torch.tensor(coord) if org_type == np.ndarray else coord
    rot = coord2rotation(coord)
    trans = coord2translation(coord)
    trans, rot = (trans.cpu().numpy(), rot.cpu().numpy()) if org_type == np.ndarray else (trans, rot)
    return trans, rot


def transquat_to_coord(trans, quat):
    return frame_to_coord((trans, quaternion_to_matrix(quat)))


def frame_to_coord(frame: tuple, unit: str='NCAC'):
    if unit == 'NCAC':
        local = UNIT_NCAC
    elif unit == 'CACN':
        local = UNIT_CACN
    trans, rot = frame
    org_type = type(trans)
    trans = torch.tensor(trans) if org_type == np.ndarray else trans
    rot = torch.tensor(rot) if org_type == np.ndarray else rot
    trans = trans.unsqueeze(0) if len(trans.shape)==2 else trans
    rot = rot.unsqueeze(0) if len(rot.shape)==3 else rot
    local = torch.tensor(local) if type(local) == np.ndarray else local
    local = local.to(trans.device)
    coord = torch.einsum('b l c r, a r -> b l a c', rot, local)
    coord = coord + repeat(trans, 'b l c -> b l a c', a=local.shape[-2])
    if unit == 'CACN':
        coord_flat = rearrange(coord, 'b l a c -> b (l a) c')
        N = _zmat2xyz(LENGTH_N_CA, ANGLE_N_CA_C, TORSION_N, coord_flat[:,3], coord_flat[:,1], coord_flat[:,0])
        CA = _zmat2xyz(LENGTH_N_CA, ANGLE_C_N_CA, TORSION_CA, coord_flat[:,-4], coord_flat[:,-3], coord_flat[:,-1])
        C = _zmat2xyz(LENGTH_CA_C, ANGLE_N_CA_C, TORSION_C, coord_flat[:,-3], coord_flat[:,-1], CA)
        O = _zmat2xyz(LENGTH_C_O, ANGLE_CA_C_O, TORSION_PI, coord_flat[:,-1], CA, C)
        coord_flat = torch.cat([N.unsqueeze(-2), coord_flat, CA.unsqueeze(-2), C.unsqueeze(-2), O.unsqueeze(-2)], dim=-2)
        coord = rearrange(coord_flat, 'b (l a) c -> b l a c', a=4)
    coord = coord.cpu().numpy() if org_type == np.ndarray else coord
    return coord


def frame_aligned_matrix(frame):
    trans, rot = frame
    trans = trans.unsqueeze(0) if len(trans.shape)==2 else trans
    rot = rot.unsqueeze(0) if len(rot.shape)==3 else rot
    m = n = trans.shape[1]
    co_mat = repeat(trans, 'b n c -> b m n c', m=m) - repeat(trans, 'b m c -> b m n c', n=n)
    rot_inv = rearrange(rot, 'b m c1 c2 -> b m c2 c1')
    rot_inv_mat = repeat(rot_inv, 'b m r c -> b m n r c', n=n)
    return torch.einsum('b m n r c, b m n c -> b m n r', rot_inv_mat, co_mat)


def FAPE(frame1, frame2, D=10, eps=1e-8, Z=10, mean=True):
    org_type1, org_type2 = type(frame1[0]), type(frame2[0])
    frame1 = (torch.tensor(v) for v in frame1) if org_type1 == np.ndarray else frame1
    frame2 = (torch.tensor(v) for v in frame2) if org_type2 == np.ndarray else frame2
    co1_mat = frame_aligned_matrix(frame1)
    co2_mat = frame_aligned_matrix(frame2)
    device = co1_mat.device
    fape_abs = torch.sqrt(torch.pow(co1_mat - co2_mat, 2).sum(dim=-1)+eps)
    fape_clamp = torch.min(fape_abs, torch.ones_like(fape_abs).to(device)*D)
    fape = fape_clamp.mean()/Z if mean else fape_clamp.mean(dim=[1,2])/Z
    fape = fape.cpu().numpy() if org_type1 == np.ndarray else fape
    return fape




#### Functions ####
def _zmat2xyz(bond, angle, dihedral, one, two, three):
    oldvec1 = bond * torch.sin(angle) * torch.sin(dihedral)
    oldvec2 = bond * torch.sin(angle) * torch.cos(dihedral)
    oldvec3 = bond * torch.cos(angle)
    oldvec = torch.stack([oldvec1, oldvec2, oldvec3, torch.tensor(1., dtype=torch.float32)], axis=0).unsqueeze(0)
    mat = _viewat(three, two, one)
    newvec = torch.einsum('b i j, b j -> b i', mat, oldvec)
    # return
    return newvec[:,:3]

def _viewat(p1, p2, p3):
    b, *_ = p1.shape
    # vector #
    p12 = p2 - p1
    p13 = p3 - p1
    # normalize #
    z = p12 / torch.linalg.norm(p12, dim=-1, keepdim=True)
    # crossproduct #
    x = torch.cross(p13, p12)
    x /= torch.linalg.norm(x, dim=-1, keepdim=True)
    y = torch.cross(z, x)
    y /= torch.linalg.norm(y, dim=-1, keepdim=True)
    # transpation matrix
    mat = torch.stack([x, y, z, p1], dim=1).transpose(-1,-2)
    pad = repeat(torch.tensor([0,0,0,1], dtype=torch.float32).to(p1.device), '... -> b () ...', b=b)
    mat = torch.cat([mat, pad], dim=-2)
    # return
    return mat
