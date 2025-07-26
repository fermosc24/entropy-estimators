import unittest
import numpy as np
from entropy_estimators import Entropy, EntropyML, JamesSteinShrink, ChaoShen, ChaoWangJost, NBRS

class TestEntropyEstimators(unittest.TestCase):

    def setUp(self):
        self.counts_uniform = [10, 10, 10, 10]
        self.counts_skewed = [40, 1, 1, 1]
        self.counts_empty = []

    def test_entropy_ml_uniform(self):
        ent = EntropyML(self.counts_uniform)
        expected = np.log(4)
        self.assertAlmostEqual(ent, expected, places=4)

    def test_entropy_ml_skewed(self):
        ent = EntropyML(self.counts_skewed)
        self.assertTrue(ent < np.log(4))

    def test_entropy_empty(self):
        self.assertEqual(Entropy(self.counts_empty), 0.0)

    def test_entropy_mle(self):
        ent = Entropy(self.counts_uniform, method="MLE")
        expected = np.log(4)
        self.assertAlmostEqual(ent, expected, places=4)

    def test_entropy_jse(self):
        ent = Entropy(self.counts_uniform, method="JSE", K=4)
        self.assertTrue(ent > 0)

    def test_entropy_cae(self):
        ent = Entropy(self.counts_uniform, method="CAE")
        self.assertTrue(ent > 0)

    def test_entropy_cwj(self):
        ent = Entropy(self.counts_uniform, method="CWJ")
        self.assertTrue(ent > 0)

    def test_entropy_nbrs(self):
        ent = Entropy(self.counts_uniform, method="NBRS")
        self.assertTrue(ent > 0)

if __name__ == '__main__':
    unittest.main()
