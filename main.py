import numpy as np
from mltheory.gnnconv import GraphConv
from mltheory.pca import PrincipalComponentAnalysis
from sklearn.decomposition import PCA as sklearnPCA

def test_gnn():
    # example from Distill Pub
    node_labels = {6: 'A', 1: 'B', 0: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G'}
    edge_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 6]])

    num_nodes = len(node_labels.keys())
    num_feats = 5

    x = np.random.rand(num_nodes, num_feats)

    graph_conv = GraphConv(edge_idx, x, node_labels)

    weights = np.array([5, 2, 3, 4, 0, 0, 0])

    # kernel_simple = graph_conv.basic_conv(3, weights)
    kernel_cheb = graph_conv.cheb_conv(3, weights)
    print(kernel_cheb)


def test_pca():
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    p = PrincipleComponentAnalysis(2, 2, data=data)
    p.optimization_step()

    original = p.X
    transformed = p.encode()

    # Comparison to SOTA Sklearn
    pca = sklearnPCA(n_components=2)
    components = pca.fit_transform(data)

    plot = True

    if plot:
        fig = plt.figure(figsize=(14, 7))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('2 component PCA')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xlabel('Axis 1')
        ax2.set_ylabel('Axis 2')
        ax2.set_title('Original data')

        ax1.scatter(data[:, 0], data[:, 1], color='green')
        ax1.scatter(transformed[:, 0], transformed[:, 1], color='blue')
        ax2.scatter(components[:, 0], components[:, 1], color='red')

        plt.show()


if __name__ == '__main__':
    test_gnn()
