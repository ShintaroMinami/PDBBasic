from unittest import TestCase
import os
import numpy as np
from pdbbasic import readpdb, rmsd, kabsch

class TestBasicGeneration(TestCase):

    def setUp(self):
        input_file = os.path.dirname(__file__)+'/data/7bqd_notag.pdb'
        self.loopmod0 = LoopModelingSimple()
        self.loopmod = LoopModeling()
        self.struct0 = ProteinBackbone(file=input_file)
        self.pose0 = pyrosetta.pose_from_file(filename=input_file)
    
    def test_generate_without_pyrosetta(self):
        struct1 = self.loopmod0.generate(self.struct0, start=68, goal=72)
        xyz0 = self.struct0[69:73+1,0:4]
        xyz1 = struct1[69:73+1,0:4]
        rmsd = np.sqrt(np.mean(((xyz0 - xyz1)**2)))
        print("\nRMSD for test loop generation w/o pyrosetta : {:.4f} [A]  ({}-{} in neo215)".format(rmsd, 69, 73))
        self.assertLess(rmsd, 0.5) # rmsd < 0.5
