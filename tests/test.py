#! /usr/bin/env python

import sys
sys.path.append("../")
from pdbutil import ProteinBackbone as pbb
import pdbbasic as bg
import torch
import numpy as np

str1_numpy = pbb(file='7bqd.pdb').coord[:,1]
str1_torch = torch.Tensor(str1_numpy)
str2_numpy = pbb(file='7bpn.pdb').coord[:,1]
str2_torch = torch.Tensor(str2_numpy)

hoge, info = bg.readpdb('7bpn.pdb')
print(bg.writepdb(hoge))

#print(hoge-str2_numpy)
exit()


batch1 = str1_torch.unfold(0, 8, 1)[0:50].transpose(-2,-1)
batch2 = str2_torch.unfold(0, 8, 1)[0:50].transpose(-2,-1)

hoge = bg.kabsch(batch1, np.array(batch2), rot_only=True)

hoge = bg.distance_matrix(np.array(batch1))


full_numpy = pbb(file='7bqd.pdb').coord[:,0:4]
full_torch = torch.Tensor(full_numpy)

print(full_torch.shape)

batch1 = full_torch.unfold(0, 10, 1)[0:50].transpose(-2,-1).transpose(-3,-2)

torsion_torch = bg.torsion_angles(batch1)
torsion_numpy = bg.torsion_angles(np.array(batch1))
print(torsion_torch[0]/3.141592*180)
print(torsion_numpy[0]/3.141592*180)