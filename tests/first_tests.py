from classes.pss import power_spectrum
import numpy as np


def basic_test_matrix_input():
    d = [-1, -1, 0,1,2,2,2,3,4,4]
    D = np.diag(d)
    ps = power_spectrum(eigendecomposition_provided=False)
    ps.fit(D)
    f = np.eye(D.shape[0])
    evs, masses = ps.transform(f)
    assert np.all(np.abs(evs - np.array([-1,0,1,2,3,4]))< 1e-9)
    assert np.all(np.abs(masses - f)< 1e-9)