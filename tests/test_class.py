from classes.pss import power_spectrum
import numpy as np
import pytest

d = [-1, -1, 0,1,2,2,2,3,4,4]
D = np.diag(d)
f = np.eye(D.shape[0])
M = np.array([[1,1,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,0,0,0],
            [0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,1,1]])

def test_matrix_input_eigen_not_provided():
    
    ps = power_spectrum(eigendecomposition_provided=False)
    ps.fit(D)
    
    evs, masses = ps.transform(f)

    assert np.all(np.abs(evs - np.array([-1,0,1,2,3,4]))< 1e-9)
    assert np.all(np.abs(masses - M)< 1e-9)

def test_matrix_input_eigen_provided():
    
    ps = power_spectrum(eigendecomposition_provided=True)
    ps.fit(eigenvalues=d,eigenvectors=np.eye(10))
    
    evs, masses = ps.transform(f)

    assert np.all(np.abs(evs - np.array([-1,0,1,2,3,4]))< 1e-9)
    assert np.all(np.abs(masses - M)< 1e-9)

def test_catches_ordering_of_eigenvalues():

    ps = power_spectrum(eigendecomposition_provided=True)
    with pytest.raises(ValueError):
        ps.fit(eigenvalues=d[::-1],eigenvectors=np.eye(10)) #reverse order


def test_catches_non_orthonormal_a():

    ps = power_spectrum(eigendecomposition_provided=True)
    with pytest.raises(ValueError):
        ps.fit(eigenvalues=d[::-1],eigenvectors=1.1*np.eye(10)) #reverse order

def test_catches_non_orthonormal_b():

    ps = power_spectrum(eigendecomposition_provided=True)
    vs = np.eye(10)
    vs[0][0] += 1
    with pytest.raises(ValueError):
        ps.fit(eigenvalues=d[::-1],eigenvectors=vs) 

def test_different_sizes():

    ps = power_spectrum(eigendecomposition_provided=True)
    vs = np.eye(10)
    vs[0][0] += 1
    with pytest.raises(ValueError):
        ps.fit(eigenvalues=d,eigenvectors=vs[:-1,:]) 

def test_no_matrix():

    ps = power_spectrum(eigendecomposition_provided=False)

    with pytest.raises(ValueError):
        ps.fit(eigenvalues=d,eigenvectors=np.eye(10))

def test_no_evs():
    
    ps = power_spectrum(eigendecomposition_provided=True)

    with pytest.raises(ValueError):
        ps.fit(eigenvectors=np.eye(10))

def test_no_evecs():
    
    ps = power_spectrum(eigendecomposition_provided=True)

    with pytest.raises(ValueError):
        ps.fit(eigenvalues = d)