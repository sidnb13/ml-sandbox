import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from sklearn.decomposition import PCA

class PrincipleComponentAnalysis:
    def __init__(self, input_dim: int, reduction_dim: int, data: np.ndarray = None, examples: int = 10) -> None:
        # Provide 10 training examples by default in shape m * n
        self.X = data if data is not None else np.random.rand(examples, input_dim) * 10
        self.input_dim = input_dim
        self.reduction_dim = reduction_dim
        # Initialize a random pseudo-orthagonal decoding matrix
        self.D = np.random.rand(self.input_dim, self.reduction_dim)
        self.D = qr(self.D)[0][:,:self.reduction_dim]
        
    def optimization_step(self):
        # find eigendecomposition for XtX
        eigvals, eigvecs = np.linalg.eig(np.matmul(self.X.T, self.X))
        top_indices = np.argsort(-eigvals)[:self.reduction_dim]
        eigvals = eigvals[top_indices]
        eigvecs = eigvecs[:, top_indices]
        
        self.D = eigvecs
    
    def encode(self):
        return np.matmul(self.X, self.D)
        
if __name__ == '__main__':
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    p = PrincipleComponentAnalysis(2, 2, data=data)
    p.optimization_step()
    
    original = p.X
    transformed = p.encode()
    
    # Comparison to SOTA Sklearn
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    
    plot = True
    
    if plot:        
        fig = plt.figure(figsize=(14,7))
        
        ax1 = fig.add_subplot(1,2,1) 
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('2 component PCA')
        
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_xlabel('Axis 1')
        ax2.set_ylabel('Axis 2')
        ax2.set_title('Original data')
        
        ax1.scatter(data[:,0],data[:,1],color='green')
        ax1.scatter(transformed[:,0], transformed[:,1],color='blue')
        ax2.scatter(components[:,0], components[:,1], color='red')
        
        plt.show()