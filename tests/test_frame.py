from unittest import TestCase
import os
import torch
import numpy as np
import pdbbasic as pdbb

class TestBasicGeneration(TestCase):

    def setUp(self):
        input_file = os.path.dirname(__file__)+'/data/7bqd_notag.pdb'
        torch.set_default_dtype(torch.float64)
        self.coord_numpy = pdbb.readpdb(input_file, model_id=None)
        self.coord_torch = torch.tensor(pdbb.readpdb(input_file, model_id=None))

    def test_frame_numpy(self):
        frame = pdbb.coord_to_frame(self.coord_numpy)
        coord_recon = pdbb.frame_to_coord(frame)
        rmsd = np.sqrt(((coord_recon - self.coord_numpy[...,0:3,:])**2).sum(-1).mean())
        self.assertLess(rmsd.item(), 0.1)
    
    def test_frame_torch(self):
        frame = pdbb.coord_to_frame(self.coord_torch)
        coord_recon = pdbb.frame_to_coord(frame)
        rmsd = torch.sqrt(((coord_recon - self.coord_torch[...,0:3,:])**2).sum(-1).mean())
        self.assertLess(rmsd.item(), 0.1)
