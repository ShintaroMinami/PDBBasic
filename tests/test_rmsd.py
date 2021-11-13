from unittest import TestCase
import os
import torch
from einops import rearrange
import pdbbasic as pdbb

class TestBasicGeneration(TestCase):

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        input_file = os.path.dirname(__file__)+'/data/7bqd_notag.pdb'
        coord = pdbb.readpdb(input_file, backbone_only=True)[:,1]
        self.ca_batch1 = rearrange(coord, '(b l) c -> b l c', b=20)
        self.ca_batch2 = self.ca_batch1 + 10.0
    
    def test_rmsd_numpy(self):
        rmsd = pdbb.rmsd(self.ca_batch1, self.ca_batch2)
        self.assertAlmostEqual(rmsd.max(), 0)
    
    def test_rmsd_torch(self):
        ca_batch1 = torch.Tensor(self.ca_batch1)
        ca_batch2 = torch.Tensor(self.ca_batch2)
        rmsd = pdbb.rmsd(ca_batch1, ca_batch2)
        self.assertAlmostEqual(rmsd.max().item(), 0)