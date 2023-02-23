import networkx as nx
import numpy as np
from numpy import polynomial as P


class GraphConv:
    def __init__(self, edge_idx: np.ndarray, x: np.ndarray, node_labels: dict) -> None:
        # provide GraphConv fields
        self.edge_idx = edge_idx
        self.x = x
        self.num_nodes = x.shape[0]
        self.node_labels = node_labels

        # create the adjacency matrix
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for edge in self.edge_idx:
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1

        self.networkx_graph = nx.relabel_nodes(
            nx.from_numpy_matrix(self.adjacency_matrix), self.node_labels
        )

        # create degree matrix
        self.degree_matrix = np.zeros_like(self.adjacency_matrix)
        np.fill_diagonal(self.degree_matrix, np.sum(self.adjacency_matrix, axis=1))

        self.laplacian = self.degree_matrix - self.adjacency_matrix

    def basic_conv(self, degree: int, weights: np.ndarray) -> np.ndarray:
        # stack an array of laplacians horizontally given degree
        weights = weights[:degree]
        laplace_stack = np.array(
            [np.linalg.matrix_power(self.laplacian, i) for i in range(degree)]
        )
        # get the conv matrix output
        weight_stack = np.array(
            [weights[i] * np.identity(self.laplacian.shape[0]) for i in range(degree)]
        )
        return np.sum(np.matmul(laplace_stack, weight_stack), axis=0)

    def cheb_conv(self, degree: int, weights: np.ndarray) -> np.ndarray:
        # normalize the laplacian according to eigenvalue
        eigvals = np.linalg.eigvals(self.laplacian)
        laplace_norm = 2 * self.laplacian / np.max(eigvals) - np.identity(
            self.laplacian.shape[0]
        )
        kernel_stack = np.array(
            [np.identity(self.laplacian.shape[0])]
            + [
                P.chebyshev.chebval(
                    np.linalg.matrix_power(laplace_norm, i), P.Chebyshev(range(i)).coef
                )
                for i in range(1, degree)
            ]
        )
        weight_stack = np.array(
            [weights[i] * np.identity(self.laplacian.shape[0]) for i in range(degree)]
        )
        return np.sum(np.matmul(kernel_stack, weight_stack), axis=0)
