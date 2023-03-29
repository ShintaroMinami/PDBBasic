from unittest import TestCase
import os
import pickle
import torch
from einops import rearrange
import pdbbasic as pdbb

class TestBasicGeneration(TestCase):

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        input_file = os.path.dirname(__file__)+'/data/7bqd_notag.pdb'
        ref_numpy_pkl = os.path.dirname(__file__)+'/data/RmsdMat5x15_numpy.pkl'
        ref_torch_pkl = os.path.dirname(__file__)+'/data/RmsdMat5x15_torch.pkl'
        with open(input_file, 'r') as fh:
            coordtest = pdbb.readpdb(fh.read())
        coord = pdbb.readpdb(input_file, CA_only=True, model_id=None)
        self.ca_batch1 = rearrange(coord, '(b l) c -> b l c', b=20)
        self.ca_batch2 = self.ca_batch1 + 10.0
        with open(ref_numpy_pkl, 'rb') as f:
            self.rmsdmat_ref_numpy = pickle.load(f)
        with open(ref_torch_pkl, 'rb') as f:
            self.rmsdmat_ref_torch = pickle.load(f)
        torch.set_default_dtype(torch.float64)

    def test_rmsd_numpy(self):
        rmsd = pdbb.rmsd(self.ca_batch1, self.ca_batch2)
        self.assertLess(rmsd.max().item(), 0.00001)
    
    def test_rmsd_torch(self):
        ca_batch1 = torch.Tensor(self.ca_batch1)
        ca_batch2 = torch.Tensor(self.ca_batch2)
        rmsd = pdbb.rmsd(ca_batch1, ca_batch2, eps=0)
        self.assertLess(rmsd.max().item(), 0.00001)

    def test_rmsd_many_vs_many_numpy(self):
        ca_batch1 = self.ca_batch1[:5]
        ca_batch2 = self.ca_batch1[5:]
        rmat = pdbb.rmsd_many_vs_many(ca_batch1, ca_batch2)
        self.assertLess((self.rmsdmat_ref_numpy-rmat).max().item(), 0.00001)
        
    def test_rmsd_many_vs_many_torch(self):
        ca_batch1 = torch.Tensor(self.ca_batch1[:5])
        ca_batch2 = torch.Tensor(self.ca_batch1[5:])
        rmat = pdbb.rmsd_many_vs_many(ca_batch1, ca_batch2)
        self.assertLess((self.rmsdmat_ref_torch-rmat).max().item(), 0.00001)

