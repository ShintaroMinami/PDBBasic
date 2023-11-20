# PDBBasic
Basic Functions for Protein Structure Data

## Install
```
pip install pdbbasic
```

## Usage
``` python
import numpy as np
import torch
import pdbbasic as pdbb

# simple PDB file read
coord1 = pdbb.readpdb('filename.pdb')
# coord1.shape -> (N, 4, 3), N=length, 4=atoms:(N,CA,C,O), 3=coordinates:(x,y,z)

# read PDB with information (chain-id, residue-number, residue-type, occupancy, b-factor)
coord1, info1 = pdbb.readpdb('filename.pdb', with_info=True)

# read mmCIF file
coord1, info1 = pdbb.readmmcif('filename.cif', with_info=True)

# download from PDB
coord1, info1 = pdbb.download('7bqd', with_info=True)

# calc RMSD
ca1 = coord1[:,1]
ca2 = pdbb.readpdb('filename.pdb', CA_only=True)

rmsd_np = pdbb.rmsd(ca1, ca2)

# Kabsch superposition
coo_sup1, coo_sup2 = pdbb.kabsch(ca1, ca2)

# torsion angle
torsion = pdbb.torsion_angles(coord1)
# torsion.shape -> (N, 3), 3=dihedrals:(phi,psi,omega)

# distance matrix
distmat_within = pdbb.distance_matrix(ca1)
distmat_between = pdbb.distance_matrix(ca1, ca2)

# torch Tensor is applicable
rmsd_torch = pdbb.rmsd(torch.Tensor(ca1), torch.Tensor(ca2))

# Frame representation like AlphaFold (translation, rotation)
trans, rot = pdbb.coord_to_frame(coord1)
# trans.shape -> (N, 3), rot.shape -> (N, 3, 3)
frame = trans, rot
coord_recon = pdbb.frame_to_coord(frame)

# FAPE (Frame Aligned Position Error)
frame1 = pdbb.coord_to_frame(coord1)
frame2 = pdbb.coord_to_frame(coord2)
fape = pdbb.FAPE(frame1, frame2)

# batched calculation is applicable
ca_batch1 = np.repeat(np.expand_dims(ca1, axis=0), 100, axis=0)
ca_batch2 = np.repeat(np.expand_dims(ca2, axis=0), 100, axis=0)
bb_batch = np.repeat(np.expand_dims(coord1, axis=0), 100, axis=0)

rmsd_batch = pdbb.rmsd(ca_batch1, ca_batch2)
sup_batch1, sup_batch2 = pdbb.kabsch(ca_batch1, ca_batch2)
torsion_batch = pdbb.torsion_angles(bb_batch)
distmat_batch = pdbb.distance_matrix(ca_batch1)

coord_batch = np.repeat(np.expand_dims(coord1, axis=0), 100, axis=0)
frame_batch = pdbb.coord_to_frame(coord_batch)
coord_recon_batch = pdbb.frame_to_coord(frame_batch)
fape_batch = pdbb.FAPE(frame_batch, frame_batch)

# all against all RMSD calculation
rmsd_matrix = pdbb.rmsd_many_vs_many(ca_batch1)

```

## Requirement
* python3
* numpy
* pandas
* pytorch
* einops
