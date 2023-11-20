import numpy as np
import torch
from einops import rearrange, repeat
from .quaternion import quaternion_to_matrix

ANGLE_N_CA_C = torch.tensor(np.deg2rad(111.2, dtype=np.float32))
ANGLE_CA_C_N = torch.tensor(np.deg2rad(116.2, dtype=np.float32))
ANGLE_C_N_CA = torch.tensor(np.deg2rad(124.2, dtype=np.float32))
ANGLE_CA_C_O = torch.tensor(np.deg2rad(120.8, dtype=np.float32))
DEFAULT_TORSION_PHI = torch.tensor(np.deg2rad(-150, dtype=np.float32))
DEFAULT_TORSION_PSI = torch.tensor(np.deg2rad(120, dtype=np.float32))
DEFAULT_TORSION_OMEGA = torch.tensor(np.deg2rad(180, dtype=np.float32))
DEFAULT_TORSION_O = torch.tensor(np.deg2rad(-30, dtype=np.float32))
LENGTH_N_CA = 1.458
LENGTH_CA_C = 1.523
LENGTH_C_N  = 1.329
LENGTH_C_O  = 1.231

UNIT_NCAC = np.array([
        [-0.5229,  1.3598,  0.0000], # N
        [ 0.0000,  0.0000,  0.0000], # CA
        [ 1.5244,  0.0000,  0.0000]  # C
    ], dtype=np.float32)

UNIT_CACN = np.array([
        [-0.6839,  1.3617,  0.0000], # CA
        [ 0.0000,  0.0000,  0.0000], # C
        [-0.6656, -1.0349,  0.0000], # O
        [ 1.3288,  0.0000,  0.0000]  # N
    ], dtype=np.float32)


def coord2rotation(coord, eps=1e-8):
    x1 = coord[:,:,2] - coord[:,:,1]
    x1 = x1 / (repeat(torch.linalg.norm(x1, dim=-1), 'b l -> b l c', c=3)+eps)
    x2 = coord[:,:,0] - coord[:,:,1]
    x2 = x2 - torch.einsum('b l c, b l -> b l c', x1, (torch.einsum('b l c, b l c -> b l', x1, x2)))
    x2 = x2 / (repeat(torch.linalg.norm(x2, dim=-1), 'b l -> b l c', c=3)+eps)
    x3 = torch.cross(x1, x2)
    x3 = x3 / (repeat(torch.linalg.norm(x3, dim=-1), 'b l -> b l c', c=3)+eps)
    return torch.stack([x1, x2, x3], dim=-2).transpose(-1,-2)


def coord2translation(coord):
    return coord[:,:,1]


def coord_to_frame(coord: torch.Tensor, unit: str='NCAC') -> tuple:
    org_type, org_shape = type(coord), coord.shape
    coord = torch.tensor(coord) if org_type == np.ndarray else coord
    coord = coord.unsqueeze(0) if len(coord.shape) == 3 else coord
    if unit == 'CACN':
        coord = torch.stack([coord[:,:-1,1],coord[:,:-1,2], coord[:,1:,0]], dim=-2)
    rot = coord2rotation(coord).squeeze() if len(org_shape) == 3 else coord2rotation(coord)
    trans = coord2translation(coord).squeeze() if len(org_shape) == 3 else coord2translation(coord)
    trans, rot = (trans.cpu().numpy(), rot.cpu().numpy()) if org_type == np.ndarray else (trans, rot)
    return trans, rot


def transquat_to_coord(trans, quat):
    return frame_to_coord((trans, quaternion_to_matrix(quat)))


def frame_to_coord(frame: tuple, unit: str='NCAC', completion=False):
    if unit == 'NCAC':
        local = UNIT_NCAC
    elif unit == 'CACN':
        local = UNIT_CACN
    trans, rot = frame
    org_type, org_shape = type(trans), trans.shape
    trans = torch.tensor(trans) if org_type == np.ndarray else trans
    rot = torch.tensor(rot) if org_type == np.ndarray else rot
    trans = trans.unsqueeze(0) if len(trans.shape)==2 else trans
    rot = rot.unsqueeze(0) if len(rot.shape)==3 else rot
    local = torch.tensor(local) if type(local) == np.ndarray else local
    local = local.to(trans.device)
    coord = torch.einsum('b l c r, a r -> b l a c', rot, local)
    coord = coord + repeat(trans, 'b l c -> b l a c', a=local.shape[-2])
    if (unit == 'CACN') and completion:
        coord_flat = rearrange(coord, 'b l a c -> b (l a) c')
        N = _zmat2xyz(LENGTH_N_CA, ANGLE_N_CA_C, DEFAULT_TORSION_PSI, coord_flat[:,3], coord_flat[:,1], coord_flat[:,0], device=coord_flat.device)
        CA = _zmat2xyz(LENGTH_N_CA, ANGLE_C_N_CA, DEFAULT_TORSION_OMEGA, coord_flat[:,-4], coord_flat[:,-3], coord_flat[:,-1], device=coord_flat.device)
        C = _zmat2xyz(LENGTH_CA_C, ANGLE_N_CA_C, DEFAULT_TORSION_PHI, coord_flat[:,-3], coord_flat[:,-1], CA, device=coord_flat.device)
        O = _zmat2xyz(LENGTH_C_O, ANGLE_CA_C_O, DEFAULT_TORSION_O, coord_flat[:,-1], CA, C, device=coord_flat.device)
        coord_flat = torch.cat([N.unsqueeze(-2), coord_flat, CA.unsqueeze(-2), C.unsqueeze(-2), O.unsqueeze(-2)], dim=-2)
        coord = rearrange(coord_flat, 'b (l a) c -> b l a c', a=4)
    elif unit == 'CACN': # [3:-1] to extracts NCACO format from CACON, like "CA C O | N ... CA C O | N"
        coord_flat = rearrange(coord, 'b l a c -> b (l a) c')[:,3:-1]
        coord = rearrange(coord_flat, 'b (l a) c -> b l a c', a=4)
    coord = coord.squeeze() if len(org_shape) == 2 else coord
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


def FAPE(frame1, frame2, weight=None, D=10, eps=1e-8, Z=10, mean=False):
    org_type1, org_type2 = type(frame1[0]), type(frame2[0])
    frame1 = (torch.tensor(v) for v in frame1) if org_type1 == np.ndarray else frame1
    frame2 = (torch.tensor(v) for v in frame2) if org_type2 == np.ndarray else frame2
    co1_mat = frame_aligned_matrix(frame1)
    co2_mat = frame_aligned_matrix(frame2)
    device = co1_mat.device
    fape_abs = torch.sqrt(torch.pow(co1_mat - co2_mat, 2).sum(dim=-1)+eps)
    fape_clamp = torch.min(fape_abs, torch.ones_like(fape_abs).to(device)*D)
    if weight is not None:
        fape_clamp = fape_clamp * weight
        weight_mean = weight.mean() if mean else weight.mean(dim=[1,2])
    else:
        weight_mean = 1.0
    fape = fape_clamp.mean()/weight_mean/Z if mean else fape_clamp.mean(dim=[1,2])/weight_mean/Z
    fape = fape.cpu().numpy() if org_type1 == np.ndarray else fape
    return fape




#### Functions ####
def _zmat2xyz(bond, angle, dihedral, one, two, three, device):
    oldvec1 = bond * torch.sin(angle) * torch.sin(dihedral)
    oldvec2 = bond * torch.sin(angle) * torch.cos(dihedral)
    oldvec3 = bond * torch.cos(angle)
    oldvec = torch.stack([oldvec1, oldvec2, oldvec3, torch.tensor(1., dtype=torch.float32)], axis=0).unsqueeze(0).to(device)
    mat = _viewat(three, two, one)
    newvec = torch.einsum('b i j, b j -> b i', mat, oldvec)
    # return
    return newvec[:,:3].to(device)

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
