import numpy as np
from scipy.sparse import csr_array

class power_spectrum:
    '''
    Numpy implementation of power spectrum computation 
    For fixed self-adjoint (n,n)-array M, and signals f as an n-dim vector, obtains magnitude-squared of projections of f onto the distinct eigenspaces of M

    Projection matrices onto each distinct eigenspaces not explicitly computed. Projection of signals onto eigenspaces inferred from eigenvectors when transform is called

    '''

    def __init__(self, precision = 1e-9, eigendecomposition_provided = False):
        '''
        input

        precision:                      float, eigenvalues with separation less than precision are not treated as distinct
        eigendecomposition_provided:    if True, then eigenvalues and orthonormal eigenvectors provided when fit; else use matrix
    
        '''
        self.precision = np.abs(precision)
        self.eigendecomposition_provided = eigendecomposition_provided
        self.fitted_ = False

    def fit(self, mat = None, eigenvalues = None, eigenvectors = None):
        
        if self.eigendecomposition_provided : # eigendecompositions pre-computed
            if eigenvalues is None:
                raise ValueError('No eigenvalues provided.')
            if eigenvectors is None:
                raise ValueError('No eigenvectors provided.')
            eigenvalues = np.array(eigenvalues)
            if len(eigenvalues.shape) != 1:
                raise ValueError('Eigenvalues are not 1-dim array or list.')
            if len(eigenvectors.shape) != 2:
                raise ValueError('Eigenvectors not nxn array, where n = num eigenvalues.')
            l = len(eigenvalues)
            if eigenvectors.shape[0] != l or eigenvectors.shape[1] != l:
                raise ValueError('Eigenvectors not nxn array, where n = num eigenvalues.')
            if np.any(np.diff(eigenvalues) < -self.precision): 
                raise ValueError('Eigenvalues not sorted!')
            if np.any(np.abs(eigenvectors @ eigenvectors.T - np.eye(eigenvectors.shape[0])) > self.precision):
                raise ValueError('Eigenvectors not orthonormal!') 
        else:
            if mat is None:
                raise ValueError('Self-adjoint matrix not provided.')
            eigenvalues, eigenvectors = np.linalg.eigh(mat)

        self._compute_pss_from_eigendecompostion(eigenvalues,eigenvectors)
        self.fitted_ = True
    
    def _compute_pss_from_eigendecompostion(self, eigenvalues, eigenvectors):
        """
        Defines distinct eigenspaces for each distinct eigenvalue

        input
            eigenvalues: sorted (n,) array of real eigenvalues
            eigenvectors: (n,n) array of normalised eigenvectors in columns; eigenvectors[:,i] has eigenvalue[i]
        """

        #assume eigenvalues, eigenvectors sorted
        #collate eigenvalues that are below precision

        diffs = np.diff(eigenvalues, prepend = -np.inf) #prepend to make sure length n, and first entry always true            
        
        distinct_test = np.abs(diffs) > self.precision # length n, i-th entry = 1 iff i-th eigenvalue distinct from (i-1)-th eigenvalue 
        self.distinct_eigenvalues_ = eigenvalues[distinct_test] # extracts distinct eigenvalues
        self.eigenvectors_ = eigenvectors.T #encode rows as eigenvectors

        self.distinct_eigenvalue_index_ = np.cumsum(distinct_test, dtype = int)-1 # (n,) array; if j-th entry = i then j-th eigenvector in i-th eigenspace 
        

        
    def transform(self, f):
        '''
        Compute power spectrum signature of signals f_1, ..., f_m, after nhaving fitted to a fixed matrix. 

        input
            f:  size (n,m) numpy array. Each column contains a signal vector
        return 
            distinct_eigenvalues :      (k,) array of distinct eigenvalues
            masses:                     (k,m) array; for each column j, masses[i,j] is the norm of projection of signal f_j onto i-th eigenspace. 
        '''
        if not self.fitted_:
            raise AttributeError("Eigendecomposition not precomputed")
        
        n,k = f.shape[0], len(self.distinct_eigenvalues_)
        aggregation_matrix = csr_array((np.ones(n), (self.distinct_eigenvalue_index_,np.arange(n))), shape = (k,n)) #(k,n) sparse array; if (i,j) = 1 then jth eigenvector is in i-th eigenspace
       
        projection = self.eigenvectors_ @ f #projection onto eigenspace 
        projection_norm = np.abs(projection)**2 #abs in case complex eigenvectors
        masses = aggregation_matrix @ projection_norm #sums up norm-squared of projections onto eigenvectors corresponding to eigenspaces.

        return self.distinct_eigenvalues_,  masses


    def fit_transform(self, mat = None, eigenvalues = None, eigenvectors = None, f = None):
        self.fit(mat, eigenvalues, eigenvectors)
        return self.transform(f)
