
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

class RobustCSP(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_components=4, reg=0.1, log=True):
        self.n_components = n_components
        self.reg = reg  # Regularization
        self.log = log
        self.filters_ = None
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X, y):
    
        if X.ndim != 3:
            raise ValueError("X must be 3D: (n_trials, n_channels, n_times)")
        
        self.mean_ = np.mean(X, axis=(0, 2), keepdims=True)
        self.std_ = np.std(X, axis=(0, 2), keepdims=True) + 1e-10
        
        # Normalize data
        X_norm = (X - self.mean_) / self.std_
        
        # Separate classes
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes")
        
        X1 = X_norm[y == classes[0]]
        X2 = X_norm[y == classes[1]]
        
        # Check if data is sufficient
        if len(X1) < 2 or len(X2) < 2:
            print(f"Warning: Very few trials ({len(X1)} vs {len(X2)})")
            # Use identity matrix as a fallback
            self.filters_ = np.eye(X.shape[1])[:, :self.n_components]
            return self
        
        # Calculate covariance matrices with regularization
        C1 = self._compute_regularized_cov(X1)
        C2 = self._compute_regularized_cov(X2)
        
        # Solve CSP
        try:
            D, V = eigh(C1, C1 + C2 + self.reg * np.eye(C1.shape[0]))
            
            # Sort eigenvalues
            idx = np.argsort(D)[::-1]
            V = V[:, idx]
            
            # Select components
            n_comp = min(self.n_components, V.shape[1])
            comp_per_class = n_comp // 2
            
            if comp_per_class > 0:
                filters = np.hstack([
                    V[:, :comp_per_class],
                    V[:, -comp_per_class:]
                ])
            else:
                # If n_components is very small
                filters = V[:, :n_comp]
            
            self.filters_ = filters
            
        except np.linalg.LinAlgError:
            print(f"CSP failed, using identity matrix")
            self.filters_ = np.eye(X.shape[1])[:, :self.n_components]
        
        return self
    
    def _compute_regularized_cov(self, X):
        """Calculate covariance matrix with regularization"""
        n_trials, n_channels, n_times = X.shape
        
        if n_trials < 2:
            return np.eye(n_channels)
        
        # Average of covariance matrices
        cov_sum = np.zeros((n_channels, n_channels))
        for i in range(n_trials):
            # Reshape data
            data = X[i].reshape(n_channels, n_times)
            cov = np.cov(data)
            cov_sum += cov
        
        mean_cov = cov_sum / n_trials
        
        # Add regularization
        reg_cov = mean_cov + self.reg * np.eye(n_channels)
        
        return reg_cov
    
    def transform(self, X):
        """Apply CSP"""
        if self.filters_ is None:
            raise ValueError("Must fit CSP before transforming")
        
        # Normalize data
        X_norm = (X - self.mean_) / self.std_
        
        n_trials = X.shape[0]
        n_components = self.filters_.shape[1]
        
        features = np.zeros((n_trials, n_components))
        
        for i in range(n_trials):
            # Project data
            projected = np.dot(self.filters_.T, X_norm[i])
            
            # Calculate log variance
            var = np.var(projected, axis=1)
            total_var = np.sum(var) + 1e-10
            
            if self.log:
                # Apply log for stability
                normalized_var = np.log(var / total_var + 1e-10)
            else:
                normalized_var = var / total_var
            
            features[i] = normalized_var
        
        return features
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        return self.fit(X, y).transform(X)

# Test
def test_robust_csp():
    """Test Robust CSP"""
    np.random.seed(42)
    
    # Hard case: few data
    n_trials = 15
    n_channels = 12
    n_times = 321
    
    X = np.random.randn(n_trials, n_channels, n_times) * 50  # ~50 µV
    y = np.random.randint(0, 2, n_trials)
    
    # Normal CSP (will fail)
    from mne.decoding import CSP
    try:
        csp_old = CSP(n_components=4, reg=None, log=True)
        X_old = csp_old.fit_transform(X, y)
        print(f"Old CSP: Works, shape {X_old.shape}")
    except Exception as e:
        print(f"Old CSP failed: {type(e).__name__}")
    
    # Robust CSP
    csp_new = RobustCSP(n_components=4, reg=0.1, log=True)
    X_new = csp_new.fit_transform(X, y)
    print(f"Robust CSP: Works, shape {X_new.shape}")
    print(f"Filters shape: {csp_new.filters_.shape}")

if __name__ == "__main__":
    test_robust_csp()