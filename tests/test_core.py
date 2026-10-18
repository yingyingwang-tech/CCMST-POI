import unittest
import numpy as np
import torch
from ccmst.model import CCMSTPOI,objective
from ccmst.evaluation import calibration_metrics,paired_cluster_bootstrap
from ccmst.perturb import perturb
class Tests(unittest.TestCase):
    def batch(self):
        return {'poi':torch.tensor([[2,3,4,0,0],[2,5,3,4,0]]),
                'cat':torch.tensor([[2,3,4,0,0],[2,2,3,4,0]]),
                'hour':torch.tensor([[2,4,5,0,0],[2,3,4,5,0]]),
                'dow':torch.tensor([[1,1,1,0,0],[1,1,1,1,0]]),
                'coords':torch.zeros(2,5,2),'ts':torch.tensor([[1.,2.,3.,0,0],[1.,2.,3.,4.,0]]),
                'label':torch.tensor([4,5])}
    def test_model_and_backward(self):
        m=CCMSTPOI(9,6,d=16,heads=4,layers=1,windows=(2,3,5))
        out=m(self.batch());self.assertEqual(tuple(out['alpha'].shape),(2,3))
        self.assertTrue(torch.allclose(out['alpha'].sum(-1),torch.ones(2)))
        loss,_=objective(out,self.batch()['label']);loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
    def test_perturb_preserves_labels(self):
        b=self.batch();o=perturb(b,'block',.5,42)
        self.assertTrue(torch.equal(o['label'],b['label']))
    def test_calibration_unchanged_ranking(self):
        z=np.array([[.1,2.,-.4],[.5,1.2,-1.]])
        self.assertTrue(np.array_equal((z/3).argmax(1),z.argmax(1)))
        self.assertTrue(np.isfinite(calibration_metrics(z,np.array([1,1]))['nll']))
    def test_paired_identity(self):
        x=[{'user_id':'u1','instance_id':'1','recall_10':1},{'user_id':'u2','instance_id':'2','recall_10':0}]
        res=paired_cluster_bootstrap(x,x,reps=10)
        self.assertEqual(res['delta'],0)
if __name__=='__main__':unittest.main()
