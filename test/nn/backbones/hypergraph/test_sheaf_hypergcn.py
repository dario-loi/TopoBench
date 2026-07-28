"""Unit tests for SheafHyperGCN."""

import pytest
import torch

from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.hypergraph.sheaf_hypergcn import SheafHyperGCN


def _sparse_incidence(edges, num_nodes):
    """Build a sparse COO incidence matrix from a [2, nnz] index tensor.

    Parameters
    ----------
    edges : torch.Tensor
        Incidence index of shape [2, nnz] (row: nodes, col: hyperedges).
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Sparse COO incidence of shape [num_nodes, num_hyperedges].
    """
    num_edges = int(edges[1].max()) + 1
    values = torch.ones(edges.shape[1])
    return torch.sparse_coo_tensor(edges, values, (num_nodes, num_edges))


def test_SheafHyperGCN(random_graph_input):
    """Unit test for SheafHyperGCN across the three sheaf variants.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing SheafHyperGCN.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes, num_features = x.shape
    incidence = _sparse_incidence(edges_1, num_nodes)

    # Diagonal sheaf with stalk dimension d=1.
    auto_test = NNModuleAutoTest(
        [
            {
                "module": SheafHyperGCN,
                "init": {
                    "in_channels": num_features,
                    "hidden_channels": num_features,
                    "out_channels": num_features,
                    "n_layers": 2,
                    "d": 1,
                    "sheaf_type": "DiagSheafs",
                },
                "forward": (x, incidence),
                "assert_shape": x.shape,
            },
        ]
    )
    auto_test.run()

    # General and orthogonal sheaves with stalk dimension d=2.
    for sheaf_type in ["GeneralSheafs", "OrthoSheafs"]:
        auto_test = NNModuleAutoTest(
            [
                {
                    "module": SheafHyperGCN,
                    "init": {
                        "in_channels": num_features,
                        "hidden_channels": num_features,
                        "out_channels": num_features,
                        "n_layers": 2,
                        "d": 2,
                        "sheaf_type": sheaf_type,
                    },
                    "forward": (x, incidence),
                    "assert_shape": x.shape,
                },
            ]
        )
        auto_test.run()


def test_SheafHyperGCN_reset_parameters(random_graph_input):
    """Test that reset_parameters runs for every submodule.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing SheafHyperGCN.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_features = x.shape[1]

    model = SheafHyperGCN(
        in_channels=num_features,
        hidden_channels=num_features,
        out_channels=num_features,
        d=2,
        sheaf_type="DiagSheafs",
        sheaf_left_proj=True,
        dynamic_sheaf=True,
    )
    model.reset_parameters()


def test_SheafHyperGCN_invalid_sheaf_type(random_graph_input):
    """Test that an invalid sheaf type raises a ValueError.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing SheafHyperGCN.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_features = x.shape[1]

    with pytest.raises(ValueError):
        SheafHyperGCN(
            in_channels=num_features,
            hidden_channels=num_features,
            out_channels=num_features,
            sheaf_type="Invalid",
        )
