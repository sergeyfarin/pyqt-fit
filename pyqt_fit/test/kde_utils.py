import numpy as np
from .. import kde_methods
from ..utils import namedtuple
from scipy import stats

def setupClass_norm(cls):
    cls.dist = stats.norm(0, 1)
    cls.sizes = np.r_[1000:5000:5j]
    cls.vs = [cls.dist.rvs(s) for s in cls.sizes]
    cls.args = {}
    cls.grid_accuracy = 1e-5
    cls.accuracy = 1e-4
    cls.weights = [ cls.dist.pdf(v) for v in cls.vs ]
    cls.lambdas = [ 1 - ws for ws in cls.weights ]
    cls.lower = -5
    cls.upper = 5

def setUpClass_lognorm(cls):
    cls.dist = stats.lognorm(1)
    cls.sizes = np.r_[1000:5000:5j]
    cls.args = {}
    cls.vs = [ cls.dist.rvs(s) for s in cls.sizes ]
    cls.vs = [ v[v < 20] for v in cls.vs ]
    cls.xs = np.r_[0:20:1024j]
    cls.weights = [ cls.dist.pdf(v) for v in cls.vs ]
    cls.lambdas = [ 1 - ws for ws in cls.weights ]
    cls.lower = 0
    cls.upper = 20

test_method = namedtuple('test_method', ['cls', 'accuracy', 'grid_accuracy'])

methods = [ test_method(kde_methods.ReflectionMethod, 1e-5, 1e-5)
          , test_method(kde_methods.CyclicMethod, 1e-5, 1e-5)
          , test_method(kde_methods.RenormalizationMethod, 1e-5, 1e-5)
          , test_method(kde_methods.LinearCombinationMethod, 1e-1, 1e-1)
          ]

