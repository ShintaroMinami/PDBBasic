# PDBBasic
Basic Functions for Protein Structure Data

## Usage
``` python
import numpy as np
import torch
import pdbbasic as pdbb

# read PDB file
coord1, info1 = pdbb.readpdb('filename.pdb')
coord2, info2 = pdbb.readpdb('filename.pdb')
ca1 = coord1[:,1]
ca2 = coord2[:,1]

# calc RMSD
rmsd_np = pdbb.rmsd(ca1, ca2)

# Kabsch superposition
coo_sup1, coo_sup2 = pdbb.kabsch(ca1, ca2)

# torsion angle
torsion = pdbb.torsion_angles(coord1)

# distance matrix
distmat_within = pdbb.distance_matrix(ca1)
distmat_between = pdbb.distance_matrix(ca1, ca2)

# torch Tensor is also applicable
rmsd_torch = pdbb.rmsd(torch.Tensor(ca1), torch.Tensor(ca2))

# batched calculation is applicable
ca_batch1 = np.repeat(np.expand_dims(ca1, axis=0), 100, axis=0)
ca_batch2 = np.repeat(np.expand_dims(ca2, axis=0), 100, axis=0)
bb_batch = np.repeat(np.expand_dims(coord1, axis=0), 100, axis=0)

rmsd_batch = pdbb.rmsd(ca_batch1, ca_batch2)
sup_batch1, sup_batch2 = pdbb.kabsch(ca_batch1, ca_batch2)
torsion_batch = pdbb.torsion_angles(bb_batch)
distmat_batch = pdbb.distance_matrix(ca_batch1)

```

## Requirement
* python3
* numpy
* pandas
* pytorch
* einops

