import numpy as np
import torch

def quaternion_to_matrix_numpy(
    quat: np.ndarray,
    normalize: bool = True,
    epsilon: float = 1e-8
    ) -> np.ndarray:
    if normalize:
      inv_norm = 1.0/np.sqrt(np.maximum(epsilon, (quat**2).sum(-1)))
      quat = quat * inv_norm[:, None]
    w, x, y, z = np.split(quat, [1,2,3], axis=-1)
    xx = 1 - 2 * (np.square(y) + np.square(z))
    xy = 2 * (x * y - w * z)
    xz = 2 * (x * z + w * y)
    yx = 2 * (x * y + w * z)
    yy = 1 - 2 * (np.square(x) + np.square(z))
    yz = 2 * (y * z - w * x)
    zx = 2 * (x * z - w * y)
    zy = 2 * (y * z + w * x)
    zz = 1 - 2 * (np.square(x) + np.square(y))
    rot = np.stack([
        np.concatenate([xx, xy, xz], axis=-1),
        np.concatenate([yx, yy, yz], axis=-1),
        np.concatenate([zx, zy, zz], axis=-1),
    ], axis=-2)
    return rot


def quaternion_to_matrix_torch(
    quat: torch.tensor,
    normalize: bool = True,
    epsilon: float = 1e-8
    ) -> torch.tensor:
    if normalize:
      inv_norm = 1.0/torch.sqrt((quat**2).sum(-1)+epsilon)
      quat = quat * inv_norm[:, None]
    w, x, y, z = torch.split(quat, [1,1,1,1], dim=-1)
    xx = 1 - 2 * (y*y + z*z)
    xy = 2 * (x * y - w * z)
    xz = 2 * (x * z + w * y)
    yx = 2 * (x * y + w * z)
    yy = 1 - 2 * (x*x + z*z)
    yz = 2 * (y * z - w * x)
    zx = 2 * (x * z - w * y)
    zy = 2 * (y * z + w * x)
    zz = 1 - 2 * (x*x + y*y)
    rot = torch.stack([
        torch.cat([xx, xy, xz], dim=-1),
        torch.cat([yx, yy, yz], dim=-1),
        torch.cat([zx, zy, zz], dim=-1),
    ], dim=-2)
    return rot


def quaternion_to_matrix(quat):
    if torch.is_tensor(quat):
        return quaternion_to_matrix_torch(quat)
    else:
        return quaternion_to_matrix_numpy(quat)
