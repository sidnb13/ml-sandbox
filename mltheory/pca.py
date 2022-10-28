import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

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
        # Simplify trace instead of Frobenius norm
        trace_arg = np.matmul(np.matmul(self.D.T, self.X.T), np.matmul(self.X, self.D))
        trace_arg = trace_arg / np.linalg.norm(trace_arg, axis=1)
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
    
    def plot(examples, i=0):
        x = examples[:,0]
        y = examples[:,1]
        
        if examples.shape[1] == 3:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x, y, examples[:, 2],color='red')
        else:
            plt.scatter(x,y,color='red')
        
        plt.figure(i)

    
    plot(original, 0)
    plot(transformed, 1)
    
    plt.show()