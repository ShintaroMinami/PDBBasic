# PDBBasic
Basic Functions for Protein Structure Data

## Usage
``` python
import numpy as np
import torch
import pdbbasic as pdbb


# read PDB file
coord1, info1 = pdbb.readpdb('filename1.pdb')
coord2, info2 = pdbb.readpdb('filename2.pdb')

# calc RMSD
rmsd_np = pdbb.rmsd(coord1, coord2)

# Kabsch superposition
coo_sup1, coo_sup2 = pdbb.kabsch(coord1, coord2)

# torsion angle
torsion = pdbb.torsion_angles(coord1)

# distance matrix
distmat_within = pdbb.distance_matrix(coord1)
distmat_between = pdbb.distance_matrix(coord1, coord2)

# torch Tensor is also applicable
rmsd_torch = pdbb.rmsd(torch.Tensor(coord1), torch.Tensor(coord2))

# batched calculation is applicable
coo_batch1 = np.repeat(np.expand_dims(coord1, axis=0), 100, axis=0)
coo_batch2 = np.repeat(np.expand_dims(coord2, axis=0), 100, axis=0)

rmsd_batch = pdbb.rmsd(coo_batch1, coo_batch2)
sup_batch1, sup_batch2 = pdbb.kabsch(coo_batch1, coo_batch2)
torsion_batch = pdbb.torsion_angles(coo_batch1)
distmat_batch = pdbb.distance_matrix(coo_batch1)

```

## Requirement
* python3
* numpy
* pandas
* pytorch
* einops

