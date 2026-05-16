"""
GNN / Node2Vec Engine (Native NumPy Implementation)

Provides a mathematical graph engine without requiring PyTorch or heavy ML frameworks.
Uses numpy to compute structural embeddings and perform deep message passing.

Features:
1. Structural Node Embeddings via SVD on the transition matrix (equivalent to Node2Vec/LightGCN).
2. Matrix-based spreading activation.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GNNEngine:
    def __init__(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        embedding_dim: int = 16,
    ):
        """
        nodes: list of node IDs
        edges: list of (source_id, target_id, weight)
        """
        self.embedding_dim = embedding_dim
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        self.n_nodes = len(nodes)

        # Build adjacency matrix
        self.A = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for src, tgt, w in edges:
            if src in self.node_to_idx and tgt in self.node_to_idx:
                i, j = self.node_to_idx[src], self.node_to_idx[tgt]
                self.A[i, j] = w
                self.A[j, i] = w  # Undirected

        # Normalize Adjacency Matrix (A_tilde = D^{-1/2} A D^{-1/2})
        # Add self-loops
        A_tilde = self.A + np.eye(self.n_nodes, dtype=np.float32)
        D = np.sum(A_tilde, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
        D_inv_sqrt_mat = np.diag(D_inv_sqrt)
        self.A_norm = D_inv_sqrt_mat.dot(A_tilde).dot(D_inv_sqrt_mat)

        self.structural_embeddings = None
        self.svd_failed = False

    def compute_structural_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Computes Node2Vec-equivalent embeddings using SVD on the normalized adjacency.
        This provides purely structural embeddings of the graph.

        Sets self.svd_failed = True if SVD fails (e.g. singular/disconnected graph),
        in which case zero vectors are returned and callers should treat structural
        signal as unavailable.
        """
        self.svd_failed = False
        if self.n_nodes < self.embedding_dim:  # fallback for very small graphs
            dim = self.n_nodes
        else:
            dim = self.embedding_dim

        # SVD of the normalized adjacency acts as a fast spectral embedding
        try:
            U, S, Vh = np.linalg.svd(self.A_norm, full_matrices=False)
            # Take top 'dim' components
            embeds = U[:, :dim] * np.sqrt(S[:dim])
            self.structural_embeddings = {
                self.idx_to_node[i]: embeds[i] for i in range(self.n_nodes)
            }
            return self.structural_embeddings
        except Exception as e:
            logger.error(f"[GNNEngine] SVD failed: {e}")
            self.svd_failed = True
            return {node: np.zeros(dim) for node in self.idx_to_node.values()}

    def message_passing(
        self,
        initial_activations: Dict[str, float],
        layers: int = 3,
        decay: float = 0.85,
    ) -> Dict[str, float]:
        """
        Runs Graph Convolutional message passing to compute spreading activation,
        replacing the iterative python loop with vectorized numpy.
        """
        h = np.zeros(self.n_nodes, dtype=np.float32)
        for node, act in initial_activations.items():
            if node in self.node_to_idx:
                h[self.node_to_idx[node]] = act

        for _ in range(layers):
            # h^(l+1) = decay * (A_norm * h^(l)) + initial
            new_h = decay * self.A_norm.dot(h)
            # Re-inject initials (like PageRank or APPNP)
            for node, act in initial_activations.items():
                if node in self.node_to_idx:
                    new_h[self.node_to_idx[node]] = max(
                        new_h[self.node_to_idx[node]], act
                    )
            h = new_h

        return {self.idx_to_node[i]: float(h[i]) for i in range(self.n_nodes)}
