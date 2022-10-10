import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
            self.adjacency_matrix[edge[0],edge[1]] = 1
            self.adjacency_matrix[edge[1],edge[0]] = 1
            
        self.networkx_graph = nx.relabel_nodes(nx.from_numpy_matrix(self.adjacency_matrix), self.node_labels)
        
        # create degree matrix
        self.degree_matrix = np.zeros_like(self.adjacency_matrix)
        np.fill_diagonal(self.degree_matrix, np.sum(self.adjacency_matrix, axis=1))
        
        self.laplacian = self.degree_matrix - self.adjacency_matrix
        
    def basic_poly_filter(self, degree: int, weights: np.ndarray) -> np.ndarray:
        # stack an array of laplacians horizontally given degree
        weights = weights[:degree]
        laplace_stack = np.array([np.linalg.matrix_power(self.laplacian, i) for i in range(degree)])
        # get the conv matrix output
        weight_stack = np.array([weights[i] * np.identity(self.laplacian.shape[0]) for i in range(degree)])
        return np.sum(np.matmul(laplace_stack, weight_stack), axis=0)
        
        
if __name__ == '__main__':
    # example from Distill Pub
    node_labels = {6:'A',1:'B',0:'C',2:'D',3:'E',4:'F',5:'G'}
    edge_idx = np.array([[0,1],[0,2],[0,3],[0,4],[0,5],[1,6]])
    
    num_nodes = len(node_labels.keys())
    num_feats = 5
    
    x = np.random.rand(num_nodes, num_feats)

    graph_conv = GraphConv(edge_idx, x, node_labels)
    kernel = graph_conv.basic_poly_filter(3, np.array([5, 2, 3, 4, 0, 0, 0]))
    print(xprime)