import numpy as np
import torch


def rmsd_many_vs_many(A, B=None, no_grad=True):
    B = B if B is not None else A
    (nA,a,c), (nB,_,_) = A.shape, B.shape
    if torch.is_tensor(A) == torch.is_tensor(B) == True:
        A_ext = A.unsqueeze(0).repeat(nB, 1, 1, 1).transpose(0,1).reshape(nA*nB,a,c)
        B_ext = B.unsqueeze(0).repeat(nA, 1, 1, 1).reshape(nA*nB,a,c)
        rmsd_matrix = rmsd(A_ext, B_ext).reshape(nA,nB)
    else:
        A_ext = np.repeat(np.expand_dims(A, axis=0), nB, axis=0)
        A_ext = np.transpose(A_ext, (1,0,2,3)).reshape((nA*nB,a,c))
        B_ext = np.repeat(np.expand_dims(B, axis=0), nA, axis=0).reshape((nA*nB,a,c))
        rmsd_matrix = rmsd(A_ext, B_ext).reshape((nA,nB))
    return rmsd_matrix


def rmsd_many_vs_one(many, one, no_grad=True):
    if len(many.shape) < len(one.shape):
        many, one = one, many
    if torch.is_tensor(one):
        one_extend = one.unsqueeze(0).repeat(many.shape[0], 1, 1)
    else:
        one_extend = np.repeat(np.expand_dims(one, axis=0), many.shape[0], axis=0)
    return rmsd(many, one_extend, no_grad=no_grad)


def kabsch(A, B, rot_only=False, no_grad=True):
    if torch.is_tensor(A) == torch.is_tensor(B) == True:    
        return kabsch_torch(A, B, rot_only=rot_only, no_grad=no_grad)
    else:
        return kabsch_numpy(np.array(A), np.array(B), rot_only=rot_only)


def rmsd(A, B, no_grad=True, eps=1e-10):
    if torch.is_tensor(A) == torch.is_tensor(B) == True:
        return rmsd_torch(A, B, no_grad=no_grad, eps=eps)
    else:
        return rmsd_numpy(np.array(A), np.array(B))


def kabsch_numpy(A, B, rot_only=False):
    X = np.expand_dims(A, axis=0) if len(A.shape) == 2 else A
    Y = np.expand_dims(B, axis=0) if len(B.shape) == 2 else B
    X = X - X.mean(axis=-2, keepdims=True)
    Y = Y - Y.mean(axis=-2, keepdims=True)
    C = np.einsum('b i j, b j k -> b i k', X.transpose(0,2,1), Y)
    V, _, W = np.linalg.svd(C)
    # det sign for direction correction
    d = np.sign(np.linalg.det(V) * np.linalg.det(W))
    V[:,:,-1] = V[:,:,-1] * np.repeat(np.expand_dims(d, axis=1), 3, axis=1)
    # calc rotation
    R = np.einsum('b i j, b j k -> b i k', V, W)
    # return
    if rot_only:
        return R.squeeze()
    else:
        X = np.einsum('b a c, b c r -> b a r', X, R)
        return X.squeeze(), Y.squeeze()


def kabsch_torch(A, B, rot_only=False, no_grad=True):
    X, Y = (A.detach(), B.detach()) if no_grad == True else (A, B)
    X = X.unsqueeze(0) if len(X.shape) == 2 else X
    Y = Y.unsqueeze(0) if len(Y.shape) == 2 else Y
    X = X - X.mean(dim=-2, keepdim=True)
    Y = Y - Y.mean(dim=-2, keepdim=True)
    C = torch.matmul(X.transpose(-2,-1), Y)
    # suggested from lucidrain's code
    if int(torch.__version__.split(".")[1]) < 8:
        V, _, W = torch.svd(C)
        W = W.transpose(-2,-1)
    else:
        V, _, W = torch.linalg.svd(C)
    # det sign for direction correction
    d = torch.sign(torch.det(V) * torch.det(W))
    V[:,:,-1] = V[:,:,-1] * d.unsqueeze(-1).repeat(1,3)
    # calc rotation
    R = torch.einsum('b i j, b j k -> b i k', V, W)
    # return
    if rot_only:
        return R.squeeze()
    else:
        X = torch.einsum('b a c, b c r -> b a r', X, R)
        return X.squeeze(), Y.squeeze()


def rmsd_numpy(A, B):
    X, Y = kabsch_numpy(A, B, rot_only=False)
    return np.sqrt(((X-Y)**2).sum(axis=-1).mean(axis=-1))


def rmsd_torch(A, B, no_grad=True, eps=1e-10):
    X, Y = kabsch_torch(A, B, rot_only=False, no_grad=no_grad)
    return torch.sqrt(((X-Y)**2+eps).sum(dim=-1).mean(dim=-1))

