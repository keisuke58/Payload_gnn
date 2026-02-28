# -*- coding: utf-8 -*-
"""
Defect-Centric Sub-graph Sampler for GNN-SHM

Extracts sub-graphs centered on defect nodes to address extreme class
imbalance (defect 0.06% vs healthy 99.94%). Uses k_hop_subgraph from
PyG (pure PyTorch, no torch-sparse dependency).

Usage:
    from subgraph_sampler import DefectCentricSampler
    sampler = DefectCentricSampler(num_hops=4, healthy_ratio=5)
    subgraphs = sampler.sample(graph)
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph


class DefectCentricSampler:
    """
    Extract sub-graphs centered on defect nodes for balanced training.

    For each graph:
      1. Find all defect nodes (y > 0)
      2. Expand to k-hop neighborhood (covers GNN receptive field)
      3. Optionally add random healthy nodes for context
      4. Return sub-graph(s) as PyG Data objects

    For healthy-only graphs, returns a small random sub-graph.
    """

    def __init__(self, num_hops=4, healthy_ratio=5, max_healthy_subgraph=500,
                 seed=None):
        """
        Args:
            num_hops: Number of hops for k-hop subgraph expansion.
                      Should match GNN depth (4 layers → 4 hops).
            healthy_ratio: Number of extra random healthy nodes per defect
                           node to include (for boundary learning).
            max_healthy_subgraph: Max nodes for healthy-only graph subgraph.
            seed: Random seed for reproducibility.
        """
        self.num_hops = num_hops
        self.healthy_ratio = healthy_ratio
        self.max_healthy_subgraph = max_healthy_subgraph
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        else:
            self.rng = None

    def sample(self, graph):
        """
        Sample sub-graph(s) from a single PyG Data object.

        Args:
            graph: PyG Data with x, edge_index, edge_attr, y, pos, etc.

        Returns:
            list[Data]: One or more sub-graphs.
        """
        defect_mask = graph.y > 0
        n_defect = defect_mask.sum().item()

        if n_defect == 0:
            return [self._sample_healthy_subgraph(graph)]

        return [self._sample_defect_subgraph(graph, defect_mask)]

    def _sample_defect_subgraph(self, graph, defect_mask):
        """Extract sub-graph around all defect nodes."""
        defect_nodes = defect_mask.nonzero(as_tuple=True)[0]

        # k-hop expansion from defect nodes
        node_idx, new_edge_index, mapping, edge_mask = k_hop_subgraph(
            defect_nodes,
            num_hops=self.num_hops,
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            relabel_nodes=True,
        )

        # Add extra random healthy nodes for context
        n_extra = int(n_defect_in_subgraph := defect_nodes.numel()) * self.healthy_ratio
        if n_extra > 0:
            # Nodes not already in subgraph
            in_subgraph = torch.zeros(graph.num_nodes, dtype=torch.bool)
            in_subgraph[node_idx] = True
            healthy_candidates = (~in_subgraph & ~defect_mask).nonzero(as_tuple=True)[0]

            if healthy_candidates.numel() > 0:
                n_sample = min(n_extra, healthy_candidates.numel())
                perm = torch.randperm(healthy_candidates.numel(),
                                      generator=self.rng)[:n_sample]
                extra_nodes = healthy_candidates[perm]

                # Merge extra nodes into subgraph
                all_nodes = torch.cat([node_idx, extra_nodes])
                all_nodes = torch.unique(all_nodes, sorted=True)

                # Re-extract subgraph with expanded node set
                sub_edge_index, sub_edge_attr = subgraph(
                    all_nodes, graph.edge_index, graph.edge_attr,
                    relabel_nodes=True, num_nodes=graph.num_nodes,
                )
                node_idx = all_nodes
                new_edge_index = sub_edge_index
                edge_mask = None  # edge_attr already handled by subgraph()
            else:
                sub_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None
                return self._build_data(graph, node_idx, new_edge_index, sub_edge_attr)

        if edge_mask is not None:
            sub_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None
        else:
            sub_edge_attr = sub_edge_attr if 'sub_edge_attr' in dir() else None

        return self._build_data(graph, node_idx, new_edge_index, sub_edge_attr)

    def _sample_healthy_subgraph(self, graph):
        """Sample a small random subgraph from a healthy-only graph."""
        n = graph.num_nodes
        n_sample = min(self.max_healthy_subgraph, n)
        perm = torch.randperm(n, generator=self.rng)[:n_sample]
        node_idx = perm.sort().values

        sub_edge_index, sub_edge_attr = subgraph(
            node_idx, graph.edge_index, graph.edge_attr,
            relabel_nodes=True, num_nodes=n,
        )
        return self._build_data(graph, node_idx, sub_edge_index, sub_edge_attr)

    def _build_data(self, graph, node_idx, edge_index, edge_attr):
        """Build a PyG Data from subset of nodes."""
        data = Data(
            x=graph.x[node_idx],
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=graph.y[node_idx],
        )
        if hasattr(graph, 'pos') and graph.pos is not None:
            data.pos = graph.pos[node_idx]
        return data
