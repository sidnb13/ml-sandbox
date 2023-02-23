import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr


class PrincipalComponentAnalysis:
    def __init__(
        self,
        input_dim: int,
        reduction_dim: int,
        data: np.ndarray = None,
        examples: int = 10,
    ) -> None:
        # Provide 10 training examples by default in shape m * n
        self.X = data if data is not None else np.random.rand(examples, input_dim) * 10
        self.input_dim = input_dim
        self.reduction_dim = reduction_dim
        # Initialize a random pseudo-orthagonal decoding matrix
        self.D = np.random.rand(self.input_dim, self.reduction_dim)
        self.D = qr(self.D)[0][:, : self.reduction_dim]

    def optimization_step(self):
        # find eigendecomposition for XtX
        eigvals, eigvecs = np.linalg.eig(np.matmul(self.X.T, self.X))
        top_indices = np.argsort(-eigvals)[: self.reduction_dim]
        eigvals = eigvals[top_indices]
        eigvecs = eigvecs[:, top_indices]

        self.D = eigvecs

    def encode(self):
        return np.matmul(self.X, self.D)
