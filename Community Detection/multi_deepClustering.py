import sys
import os
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator, eigsh

from sklearn.cluster import KMeans
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition

import scipy.sparse as sp
from collections import defaultdict

import argparse
import numpy as np
import scipy.sparse
from scipy.sparse import base
import sklearn.metrics
import tensorflow.compat.v2 as tf
from tools import dmon
from tools import gcn
from tools import metrics
from tools import utils
from tools import multi_dmon  # <-- multi-group DMoN implementation

tf.compat.v1.enable_v2_behavior()




def load_graphMultiGroupDiversity(
    edgelist_file,
    features_file,
    featuresType,
    group_values=None,
    drop_zero_edge_groups=False,
):
    """
    Group-only diversity loader.

    Builds:
      - full adjacency A (n x n),
      - group-vs-rest diversity adjacencies A_g^{div} for each group g,
        keeping edges with exactly one endpoint in g (XOR),
      - feature matrix,
      - node attribute dict/df.

    Returns
    -------
    adjacency : sp.csr_matrix (n x n)
        Symmetrized full adjacency.
    group_div_adj_dict : dict
        Maps group g -> A_g^{div} (sp.csr_matrix, n x n).
    groups : list
        Ordered list of groups actually returned (keys of group_div_adj_dict).
    features : sp.csr_matrix
        Feature matrix.
    node_attribute_dict : dict
        node_idx -> attribute.
    node_attribute_df : pd.DataFrame
        columns ['node_idx','attribute'].
    """

    # --- 1. Load edges and build adjacency ---
    edges = pd.read_csv(edgelist_file, sep=" ", header=None)
    nodes = pd.concat([edges[0], edges[1]]).unique()
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    row = edges[0].map(node_to_idx).to_numpy()
    col = edges[1].map(node_to_idx).to_numpy()

    adjacency = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.float32), (row, col)),
        shape=(len(nodes), len(nodes)),
    ).tocsr()

    # undirected, unweighted 0/1
    adjacency = (adjacency + adjacency.T).minimum(1)

    # --- 2. Load attributes and map to node indices ---
    features_data = pd.read_csv(features_file)
    features_data["node_idx"] = features_data["nodes"].map(node_to_idx)
    features_data = features_data.dropna(subset=["node_idx"])
    features_data["node_idx"] = features_data["node_idx"].astype(int)

    node_attribute_dict = dict(zip(features_data["node_idx"], features_data["attribute"]))
    node_attribute_df = pd.DataFrame(
        {"node_idx": features_data["node_idx"], "attribute": features_data["attribute"]}
    )

    # --- 3. Determine groups ---
    if group_values is None:
        candidate_groups = sorted(features_data["attribute"].unique().tolist(), key=lambda x: str(x))
    else:
        present = set(features_data["attribute"].unique().tolist())
        candidate_groups = sorted([g for g in group_values if g in present], key=lambda x: str(x))

    if len(candidate_groups) < 2:
        raise ValueError(
            "load_graphMultiGroupDiversity_GroupOnly: need at least two distinct groups "
            f"(got {candidate_groups})."
        )

    # --- 4. Precompute group ids (fast XOR masking) ---
    n = len(nodes)
    group_to_gid = {g: i for i, g in enumerate(candidate_groups)}

    gid = np.full(n, -1, dtype=np.int32)
    for i, a in node_attribute_dict.items():
        if a in group_to_gid:
            gid[i] = group_to_gid[a]

    coo = adjacency.tocoo()
    u = coo.row
    v = coo.col
    data = coo.data

    gu = gid[u]
    gv = gid[v]
    valid = (gu >= 0) & (gv >= 0)

    # --- 5. Build A_g^{div} for each group g (XOR: exactly one endpoint in g) ---
    group_div_adj_dict = {}
    groups = []

    for g in candidate_groups:
        gid_g = group_to_gid[g]
        mask = valid & np.logical_xor(gu == gid_g, gv == gid_g)

        if not mask.any():
            if drop_zero_edge_groups:
                continue
            A_g = sp.coo_matrix((n, n), dtype=np.float32).tocsr()
        else:
            A_g = sp.coo_matrix((data[mask], (u[mask], v[mask])), shape=(n, n)).tocsr()

        group_div_adj_dict[g] = A_g
        groups.append(g)

    if len(groups) == 0:
        raise ValueError(
            "load_graphMultiGroupDiversity_GroupOnly: no group had any cross-to-rest edges "
            "(or all were dropped)."
        )

    # --- 6. Build features (same logic as your loaders) ---
    feature_rows = features_data["node_idx"].astype(int).to_numpy()

    if featuresType == "degree":
        # NOTE: adjacency is symmetric so coo contains both directions; consistent with your previous loaders
        degrees = defaultdict(int)
        for uu, vv in zip(u, v):
            degrees[uu] += 1
            degrees[vv] += 1
        values = np.array([degrees[i] for i in feature_rows], dtype=np.float32)
        features = sp.csr_matrix((values, (feature_rows, feature_rows)), shape=(n, n))

    elif featuresType == "id":
        features = sp.csr_matrix((feature_rows, (feature_rows, feature_rows)), shape=(n, n))

    else:
        attributes = features_data["attribute"].to_numpy()
        features = sp.csr_matrix(
            (attributes, (feature_rows, np.zeros_like(feature_rows))),
            shape=(n, 1),
        )

    return adjacency, group_div_adj_dict, groups, features, node_attribute_dict, node_attribute_df


def load_graphMultiPairDiversity(edgelist_file,
                             features_file,
                             featuresType,
                             group_values=None,
                             drop_zero_edge_pairs=True):
    """
    Multi-group diversity loader.

    Builds:
      - the full adjacency A (n x n),
      - pairwise diversity adjacencies A_{g,h}^{div} for each unordered pair (g,h),
        keeping only edges that connect group g to group h,
      - standard feature matrix (same logic as the other loaders),
      - node-attribute structures.

    Parameters
    ----------
    edgelist_file : str
        Path to edge list file with two columns per line (u v), space-separated.
    features_file : str
        Path to CSV with at least columns: ['nodes', 'attribute'].
        'nodes' must match the node IDs in the edgelist.
    featuresType : {'degree', 'id', other}
        How to build the feature matrix:
            'degree' -> diagonal degree features
            'id'     -> diagonal node-id features
            other    -> single-column attribute features
    group_values : list or None, optional
        Optional explicit list of attribute values to treat as groups.
        If None, all distinct attribute values present in the features_file
        (after mapping to node_idx) are used.
    drop_zero_edge_pairs : bool, default True
        If True, for any pair (g,h) with no cross-group edges we simply do not
        create A_{g,h}^{div} and do not include (g,h) in the returned pairs list.
        This avoids degenerate pairs that would pin Q_min^{div}(S) at ~0 and
        cause division-by-zero in the null model.

    Returns
    -------
    adjacency : sp.csr_matrix (n x n)
        Symmetrized adjacency of the full graph.
    pair_adj_dict : dict
        Maps unordered group pair (g, h) with g < h to A_{g,h}^{div}
        (sp.csr_matrix of shape (n, n)). Only contains pairs with at least one
        cross-group edge if drop_zero_edge_pairs=True.
    pairs : list of tuple
        Ordered list of group pairs (g, h) actually used (same keys as in pair_adj_dict).
    features : sp.csr_matrix
        Feature matrix as in the original code (used later to build diagonal features).
    node_attribute_dict : dict
        Maps node_idx -> attribute value.
    node_attribute_df : pd.DataFrame
        DataFrame with columns ['node_idx', 'attribute'].
    """

    # --- 1. Load edges and adjacency ---
    edges = pd.read_csv(edgelist_file, sep=' ', header=None)
    nodes = pd.concat([edges[0], edges[1]]).unique()
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    row = edges[0].map(node_to_idx)
    col = edges[1].map(node_to_idx)
    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (row, col)),
        shape=(len(nodes), len(nodes))
    ).tocsr()

    # Make adjacency undirected and simple
    adjacency = (adjacency + adjacency.T).minimum(1)

    # --- 2. Load attributes and map to node indices ---
    features_data = pd.read_csv(features_file)
    # Map node IDs from the CSV to [0, n) indices
    features_data['node_idx'] = features_data['nodes'].map(node_to_idx)
    # Drop nodes that are not in the graph
    features_data = features_data.dropna(subset=['node_idx'])
    features_data['node_idx'] = features_data['node_idx'].astype(int)

    node_attribute_dict = dict(
        zip(features_data['node_idx'], features_data['attribute'])
    )
    node_attribute_df = pd.DataFrame({
        'node_idx': features_data['node_idx'],
        'attribute': features_data['attribute']
    })

    # --- 3. Determine which groups to use ---
    if group_values is None:
        candidate_groups = sorted(features_data['attribute'].unique().tolist())
    else:
        present_attrs = set(features_data['attribute'].unique().tolist())
        # keep only group_values that actually appear in the data
        candidate_groups = sorted([g for g in group_values if g in present_attrs])

    if len(candidate_groups) < 2:
        raise ValueError(
            "load_graphMultiPairDiversity: need at least two distinct groups "
            f"(got {candidate_groups})."
        )

    # --- 4. Build pairwise diversity adjacencies A_{g,h}^{div} ---
    coo = adjacency.tocoo()
    pair_adj_dict = {}
    pairs = []

    # Pre-fetch attributes into an array for speed
    # attr_array[i] = attribute of node i (or None if missing)
    max_idx = len(nodes)
    attr_array = [None] * max_idx
    for i, a in node_attribute_dict.items():
        if 0 <= i < max_idx:
            attr_array[i] = a

    for i, g in enumerate(candidate_groups):
        for h in candidate_groups[i+1:]:
            # Build mask_gh: edges whose endpoints are (g,h) or (h,g)
            mask_gh = []
            for u, v in zip(coo.row, coo.col):
                au = attr_array[u]
                av = attr_array[v]
                # Only count edges between g and h
                if (au == g and av == h) or (au == h and av == g):
                    mask_gh.append(True)
                else:
                    mask_gh.append(False)

            if not any(mask_gh):
                if drop_zero_edge_pairs:
                    # Skip this pair entirely: no cross-group edges
                    continue
                else:
                    # Create an explicit all-zero adjacency
                    A_gh = sp.coo_matrix(
                        ([], ([], [])),
                        shape=adjacency.shape
                    ).tocsr()
            else:
                A_gh = sp.coo_matrix(
                    (coo.data[mask_gh], (coo.row[mask_gh], coo.col[mask_gh])),
                    shape=adjacency.shape
                ).tocsr()

            pair_key = (g, h)
            pairs.append(pair_key)
            pair_adj_dict[pair_key] = A_gh

    if len(pairs) == 0:
        raise ValueError(
            "load_graphMultiPairDiversity: no cross-group pairs with edges found. "
            "Check your data or disable drop_zero_edge_pairs."
        )

    # --- 5. Build features (same logic as other loaders) ---
    feature_rows = features_data['node_idx'].astype(int).to_numpy()

    if featuresType == 'degree':
        degrees = defaultdict(int)
        for u, v in zip(coo.row, coo.col):
            degrees[u] += 1
            degrees[v] += 1
        values = np.array([degrees[i] for i in feature_rows])
        features = sp.csr_matrix(
            (values, (feature_rows, feature_rows)),
            shape=(len(nodes), len(nodes))
        )
    elif featuresType == 'id':
        features = sp.csr_matrix(
            (feature_rows, (feature_rows, feature_rows)),
            shape=(len(nodes), len(nodes))
        )
    else:
        attributes = features_data['attribute'].to_numpy()
        features = sp.csr_matrix(
            (attributes, (feature_rows, np.zeros_like(feature_rows))),
            shape=(len(nodes), 1)
        )

    return adjacency, pair_adj_dict, pairs, features, node_attribute_dict, node_attribute_df



def load_graphMultiGroup(edgelist_file,
                         features_file,
                         featuresType,
                         group_values=None,
                         drop_zero_edge_groups=True):
    """
    Multi-group version of load_graphGroup with optional filtering of zero-edge groups.

    Parameters
    ----------
    edgelist_file : str
        Path to edge list file (u v per line, space-separated).
    features_file : str
        Path to CSV with at least columns: ['nodes', 'attribute'].
    featuresType : {'degree', 'id', other}
        How to build input features (same logic as original DeepGroup code).
    group_values : list or None, optional
        Optional explicit list of attribute values to treat as groups.
        If None, all distinct attribute values in the data are considered.
    drop_zero_edge_groups : bool, default True
        If True, groups whose A_g^{grp} has no edges (nnz=0) are dropped
        from the returned group list and group adjacency dictionary.

    Returns
    -------
    adjacency : sp.csr_matrix (n x n)
        Symmetrized adjacency of the full graph.
    group_adj_dict : dict
        Maps group value g -> A_g^{grp} (sp.csr_matrix, n x n). If
        drop_zero_edge_groups=True, only contains groups with at least one edge.
    groups : list
        Ordered list of group values actually used in group_adj_dict.
    features : sp.csr_matrix
        Feature matrix as in the original code (used later to build diagonal features).
    node_attribute_dict : dict
        Maps node_idx -> attribute value.
    node_attribute_df : pd.DataFrame
        DataFrame with columns ['node_idx', 'attribute'].
    """

    # --- 1. Load edges and build base adjacency ---
    edges = pd.read_csv(edgelist_file, sep=' ', header=None)
    nodes = pd.concat([edges[0], edges[1]]).unique()
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    row = edges[0].map(node_to_idx)
    col = edges[1].map(node_to_idx)
    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (row, col)),
        shape=(len(nodes), len(nodes))
    ).tocsr()

    # Make adjacency undirected and simple
    adjacency = (adjacency + adjacency.T).minimum(1)

    # --- 2. Load attributes, map them to node indices ---
    features_data = pd.read_csv(features_file)
    # Map node IDs in the CSV to [0, n) indices used in adjacency
    features_data['node_idx'] = features_data['nodes'].map(node_to_idx)
    # Drop rows whose nodes are not in the graph (node_idx is NaN)
    features_data = features_data.dropna(subset=['node_idx'])
    features_data['node_idx'] = features_data['node_idx'].astype(int)

    node_attribute_dict = dict(
        zip(features_data['node_idx'], features_data['attribute'])
    )
    node_attribute_df = pd.DataFrame(
        {'node_idx': features_data['node_idx'],
         'attribute': features_data['attribute']}
    )

    # --- 3. Determine which groups to consider ---
    if group_values is None:
        candidate_groups = sorted(features_data['attribute'].unique().tolist())
    else:
        # Intersect with actually present attributes, to avoid nonsense
        present_attrs = set(features_data['attribute'].unique().tolist())
        candidate_groups = sorted([g for g in group_values if g in present_attrs])

    # --- 4. For each group g, build A_g^{grp} and optionally drop zero-edge ones ---
    coo = adjacency.tocoo()
    group_adj_dict = {}
    groups = []

    for g in candidate_groups:
        # Keep edges with at least one endpoint in group g
        mask_g = [
            (node_attribute_dict.get(i) == g) or (node_attribute_dict.get(j) == g)
            for i, j in zip(coo.row, coo.col)
        ]

        if not any(mask_g):
            # No edges incident to group g in the graph
            if drop_zero_edge_groups:
                # Skip this group entirely
                continue
            else:
                # Keep an all-zero adjacency for completeness
                A_g = sp.coo_matrix(
                    (coo.data[mask_g], (coo.row[mask_g], coo.col[mask_g])),
                    shape=adjacency.shape
                ).tocsr()
        else:
            A_g = sp.coo_matrix(
                (coo.data[mask_g], (coo.row[mask_g], coo.col[mask_g])),
                shape=adjacency.shape
            ).tocsr()

        groups.append(g)
        group_adj_dict[g] = A_g

    # Safety: if we ended up with no groups with edges, raise an error
    if len(groups) == 0:
        raise ValueError(
            "load_graphMultiGroup: no groups with incident edges found. "
            "Check your data or disable drop_zero_edge_groups."
        )

    # --- 5. Build features (same logic as before) ---
    feature_rows = features_data['node_idx'].astype(int).to_numpy()

    if featuresType == 'degree':
        degrees = defaultdict(int)
        for u, v in zip(coo.row, coo.col):
            degrees[u] += 1
            degrees[v] += 1
        values = np.array([degrees[i] for i in feature_rows])
        features = sp.csr_matrix(
            (values, (feature_rows, feature_rows)),
            shape=(len(nodes), len(nodes))
        )
    elif featuresType == 'id':
        features = sp.csr_matrix(
            (feature_rows, (feature_rows, feature_rows)),
            shape=(len(nodes), len(nodes))
        )
    else:
        attributes = features_data['attribute'].to_numpy()
        features = sp.csr_matrix(
            (attributes, (feature_rows, np.zeros_like(feature_rows))),
            shape=(len(nodes), 1)
        )

    return adjacency, group_adj_dict, groups, features, node_attribute_dict, node_attribute_df

def findOptimalK(A, max_eigs=100):
    n = A.shape[0]
    d = np.array(A.sum(axis=1)).flatten().astype(float)
    m = d.sum() / 2

    sqrt_d = np.sqrt(d, where=(d > 0), out=np.zeros_like(d))
    inv_sqrt = np.divide(1.0, sqrt_d, where=(sqrt_d > 0), out=np.zeros_like(d))

    def matvec(x):
        y = inv_sqrt * x
        first = inv_sqrt * (A.dot(y))
        correction = (d.dot(y) / (2 * m)) * inv_sqrt * d
        return first - correction

    B_op = LinearOperator((n, n), matvec=matvec, dtype=float)

    eigvals = eigsh(B_op, k=min(max_eigs, n - 2), which="LA", return_eigenvectors=False)
    sorted_vals = np.sort(eigvals)[::-1]
    gaps = np.diff(sorted_vals)

    idxs = np.argsort(gaps)
    return idxs[-1] + 1


def convert_scipy_sparse_to_sparse_tensor(matrix):
    matrix = matrix.tocoo()
    return tf.sparse.SparseTensor(
        np.vstack([matrix.row, matrix.col]).T,
        matrix.data.astype(np.float32),
        matrix.shape,
    )




def build_multi_group_dmon(input_features,
                           input_graph,
                           input_adjacency,
                           input_group_graphs,
                           clustersNumber,
                           lamda):
    """
    Multi-group version of build_group_dmon.

    Expects:
      - input_features: node feature matrix
      - input_graph: normalized adjacency for GCN
      - input_adjacency: original adjacency for modularity
      - input_group_graphs: list of per-group adjacency inputs A_g^{grp}
    """

    output = input_features
    for n_channels in [64]:
        num_nodes = input_graph.shape[1]
        output = gcn.GCN(num_nodes, n_channels)([output, input_graph])

    # multi_dmon.MultiGroupDMoN implements the max-min group modularity loss
    pool, pool_assignment = multi_dmon.MultiGroupDMoN(
        clustersNumber,
        collapse_regularization=1,
        dropout_rate=0.2
    )([output, input_adjacency] + input_group_graphs, lamda=lamda)

    model_inputs = [input_features, input_graph, input_adjacency] + input_group_graphs
    return tf.keras.Model(
        inputs=model_inputs,
        outputs=[pool, pool_assignment]
    )


def deepMultiGroupClustering(edgelist_path,
                        attributes_path,
                        k_opt,
                        featuresType='id',
                        lamda=0.5):
    """
    Multi-group DeepGroup clustering (max-min group modularity).

    Steps:
      1. Load graph and sensitive attribute; discover all groups U.
      2. Build per-group adjacencies A_g^{grp} for every g in U.
      3. Feed adjacency + {A_g^{grp}} into MultiGroupDMoN, which:
         - computes Q_g(S) for each group,
         - finds Q_min^{grp}(S) = min_g Q_g(S),
         - optimizes -Q(S) - lamda * Q_min^{grp}(S) + collapse regularizer.
    """

    # Load multi-group adjacencies and attributes
    adjacency, group_adj_dict, groups, features, node_attributes_dict, node_attribute_df = \
        load_graphMultiGroup(edgelist_path, attributes_path, featuresType)

    # Turn sparse feature matrix into a dense vector of node features (like original DeepGroup)
    original_features = features.copy()
    diag = original_features.diagonal().astype(np.float32)
    features = diag.reshape(-1, 1)

    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]

    # Main graph adjacency: raw + normalized
    graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(adjacency.copy())
    )

    # Per-group adjacencies A_g^{grp} (raw)
    group_sparse_graphs = []
    for g in groups:
        A_g = group_adj_dict[g]
        group_sparse_graphs.append(convert_scipy_sparse_to_sparse_tensor(A_g))

    # Keras inputs
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_group_graphs = [
        tf.keras.layers.Input((n_nodes,), sparse=True)
        for _ in groups
    ]

    # Optimal k from the original adjacency
    k_opt = findOptimalK(adjacency)
    #k_opt = 20
    # Build model with MultiGroupDMoN
    model = build_multi_group_dmon(
        input_features,
        input_graph,
        input_adjacency,
        input_group_graphs,
        k_opt,
        lamda,
    )

    def grad(model, inputs):
        with tf.GradientTape() as tape:
            _ = model(inputs, training=True)
            loss_value = sum(model.losses)
        return model.losses, tape.gradient(loss_value, model.trainable_variables)

    lr_schedule = 0.01
    if n_nodes > 10000:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.07,
            decay_steps=100,
            decay_rate=0.9
        )
    '''if n_nodes <= 4000:
        lr_schedule = 0.001'''''
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer, None)

    features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
    train_inputs = [features_tf, graph_normalized, graph] + group_sparse_graphs

    for epoch in range(1000):
        loss_values, grads = grad(model, train_inputs)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 200 == 0:
            print(
                f'epoch {epoch}, losses: ' +
                ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values])
            )

    # Get hard cluster assignments
    _, assignments = model(train_inputs, training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)

    communities_dict = {}
    for i, c in enumerate(clusters):
        communities_dict.setdefault(c, []).append(i)
    communities = list(communities_dict.values())

    return communities


def deepMultiPairDiversityClustering(edgelist_path,
                                 attributes_path,k_opt,
                                 featuresType='id',
                                 lamda=0.5,
                                 group_values=None):
    """
    Multi-group DeepDiversity clustering (max–min pairwise diversity modularity).

    This function implements the loss

        L = - Q(S) - λ_min^{div} * Q_min^{div}(S) + collapse_regularizer,

    where:
      - Q(S) is standard modularity on the full adjacency A,
      - for each unordered pair (g,h) of groups we build A_{g,h}^{div},
      - Q_{g,h}^{div}(S) is the modularity on A_{g,h}^{div},
      - Q_min^{div}(S) = min_{(g,h)} Q_{g,h}^{div}(S),
      - λ_min^{div} = lamda controls the strength of the min-pair term.

    Steps:
      1. Load graph and attributes; discover all groups (or use group_values).
      2. Build pairwise diversity adjacencies A_{g,h}^{div} for each unordered
         pair (g,h) with at least one cross-group edge.
      3. Feed adjacency + {A_{g,h}^{div}} into MultiGroupDMoN, which:
         - computes Q_{g,h}^{div}(S) for each pair,
         - finds Q_min^{div}(S) = min_{(g,h)} Q_{g,h}^{div}(S),
         - optimizes -Q(S) - λ_min^{div} * Q_min^{div}(S) + collapse term.
    """

    # 1) Load base graph + pairwise diversity adjacencies
    adjacency, pair_adj_dict, pairs, features, node_attribute_dict, node_attribute_df = \
        load_graphMultiPairDiversity(
            edgelist_file=edgelist_path,
            features_file=attributes_path,
            featuresType=featuresType,
            group_values=group_values,
            drop_zero_edge_pairs=True
        )

    # 2) Diagonal-style features (same pattern as other deep methods)
    original_features = features.copy()
    diag = original_features.diagonal().astype(np.float32)
    features = diag.reshape(-1, 1)

    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]

    # 3) Full graph adjacency: raw + normalized
    graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(adjacency.copy())
    )

    # 4) Pairwise diversity adjacencies A_{g,h}^{div} (raw, as SparseTensor)
    pair_sparse_graphs = []
    for pair in pairs:
        A_gh = pair_adj_dict[pair]
        pair_sparse_graphs.append(
            convert_scipy_sparse_to_sparse_tensor(A_gh)
        )

    # 5) Keras inputs
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_pair_graphs = [
        tf.keras.layers.Input((n_nodes,), sparse=True)
        for _ in pairs
    ]

    # 6) Choose k as in the original code
    k_opt = findOptimalK(adjacency)
    #k_opt = 20
    # 7) Build model with MultiGroupDMoN
    #    Note: MultiGroupDMoN just sees "a list of adjacencies"
    #    and applies max–min over their modularities. If those
    #    adjacencies are pairwise diversity graphs, the term becomes
    #    max–min pairwise diversity modularity.
    model = build_multi_group_dmon(
        input_features,
        input_graph,
        input_adjacency,
        input_pair_graphs,
        k_opt,
        lamda=lamda
    )

    # 8) Training loop
    def grad(model, inputs):
        with tf.GradientTape() as tape:
            _ = model(inputs, training=True)
            loss_value = sum(model.losses)
        return model.losses, tape.gradient(loss_value, model.trainable_variables)

    lr_schedule = 0.01
    if n_nodes > 10000:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.07,
            decay_steps=100,
            decay_rate=0.9
        )
    '''if n_nodes <= 4000:
        lr_schedule = 0.001'''
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer, None)

    features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
    train_inputs = [features_tf, graph_normalized, graph] + pair_sparse_graphs

    for epoch in range(1000):
        loss_values, grads = grad(model, train_inputs)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 200 == 0:
            print(
                f'epoch {epoch}, losses: ' +
                ' '.join([f'{lv.numpy():.4f}' for lv in loss_values])
            )

    # 9) Hard cluster assignments → community list
    _, assignments = model(train_inputs, training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)

    communities_dict = {}
    for i, c in enumerate(clusters):
        communities_dict.setdefault(c, []).append(i)
    communities = list(communities_dict.values())

    # Return communities + meta, similar to multi-group version
    return communities

def deepMultiGroupDiversityClustering(edgelist_path,
                                     attributes_path,
                                     k_opt,
                                     featuresType='id',
                                     lamda=0.5,
                                     group_values=None,
                                     drop_zero_edge_groups=True):
    """
    Group-based DeepDiversity clustering:
    uses per-group XOR adjacencies A_g^{div} (group vs rest, not pairwise).
    """

    adjacency, group_div_adj_dict, groups, features, node_attribute_dict, node_attribute_df = \
        load_graphMultiGroupDiversity(
            edgelist_file=edgelist_path,
            features_file=attributes_path,
            featuresType=featuresType,
            group_values=group_values,
            drop_zero_edge_groups=drop_zero_edge_groups
        )

    original_features = features.copy()
    diag = original_features.diagonal().astype(np.float32)
    features = diag.reshape(-1, 1)

    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]

    graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(adjacency.copy())
    )

    group_sparse_graphs = [
        convert_scipy_sparse_to_sparse_tensor(group_div_adj_dict[g])
        for g in groups
    ]

    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_group_graphs = [
        tf.keras.layers.Input((n_nodes,), sparse=True)
        for _ in groups
    ]
    k_opt = findOptimalK(adjacency)
    model = build_multi_group_dmon(
        input_features,
        input_graph,
        input_adjacency,
        input_group_graphs,
        k_opt,
        lamda=lamda
    )

    def grad(model, inputs):
        with tf.GradientTape() as tape:
            _ = model(inputs, training=True)
            loss_value = sum(model.losses)
        return model.losses, tape.gradient(loss_value, model.trainable_variables)

    lr_schedule = 0.01
    if n_nodes > 10000:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.07,
            decay_steps=100,
            decay_rate=0.9
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer, None)

    features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
    train_inputs = [features_tf, graph_normalized, graph] + group_sparse_graphs

    for epoch in range(1000):
        loss_values, grads = grad(model, train_inputs)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 200 == 0:
            print(
                f'epoch {epoch}, losses: ' +
                ' '.join([f'{lv.numpy():.4f}' for lv in loss_values])
            )

    _, assignments = model(train_inputs, training=False)
    clusters = assignments.numpy().argmax(axis=1)

    communities_dict = {}
    for i, c in enumerate(clusters):
        communities_dict.setdefault(c, []).append(i)

    return list(communities_dict.values())




def build_fairness_group_dmon(input_features,
                              input_graph,
                              input_adjacency,
                              input_group_graphs,
                              clustersNumber,
                              lamda):
    """
    Build the multi-group fairness-aware DMoN model.

    This wraps MultiFairnessDMoN, which:
      - takes as input node features, the full adjacency,
        and a list of per-group adjacencies A_g^{grp},
      - computes standard modularity Q(S),
      - computes group modularities Q_g(S) for each group g,
      - adds a fairness penalty proportional to 
        |max_g Q_g(S) - min_g Q_g(S)|.

    The resulting loss is:
      L = -Q(S) + lamda * |max_g Q_g(S) - min_g Q_g(S)|
          + collapse_regularization * R_collapse.
    """

    output = input_features
    for n_channels in [64]:
        num_nodes = input_graph.shape[1]
        output = gcn.GCN(num_nodes, n_channels)([output, input_graph])

    # MultiFairnessDMoN implements the multi-group modularity-gap fairness loss
    pool, pool_assignment = multi_dmon.MultiFairnessDMoN(
        clustersNumber,
        collapse_regularization=1,
        dropout_rate=0.2
    )([output, input_adjacency] + input_group_graphs, lamda=lamda)

    model_inputs = [input_features, input_graph, input_adjacency] + input_group_graphs
    return tf.keras.Model(
        inputs=model_inputs,
        outputs=[pool, pool_assignment]
    )



def deepMultiFairnessClustering(edgelist_path,
                        attributes_path,
                        k_opt,
                        featuresType='id',
                        lamda=0.5):
    """
    Multi-group DeepGroup-style clustering with group modularity gap fairness.

    Steps:
      1. Load graph and sensitive attributes; discover all groups U.
      2. Build per-group adjacencies A_g^{grp} for every g in U.
      3. Feed adjacency + {A_g^{grp}} into MultiFairnessDMoN, which:
         - computes group modularities Q_g(S) for each group g,
         - computes the gap |max_g Q_g(S) - min_g Q_g(S)|,
         - optimizes  -Q(S) + lamda * |max_g Q_g(S) - min_g Q_g(S)| 
           plus the standard collapse regularizer.
    """

    # Load multi-group adjacencies and attributes
    adjacency, group_adj_dict, groups, features, node_attributes_dict, node_attribute_df = \
        load_graphMultiGroup(edgelist_path, attributes_path, featuresType)

    # Turn sparse feature matrix into a dense vector of node features (like original DeepGroup)
    original_features = features.copy()
    diag = original_features.diagonal().astype(np.float32)
    features = diag.reshape(-1, 1)

    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]

    # Main graph adjacency: raw + normalized
    graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(adjacency.copy())
    )

    # Per-group adjacencies A_g^{grp} (raw)
    group_sparse_graphs = []
    for g in groups:
        A_g = group_adj_dict[g]
        group_sparse_graphs.append(convert_scipy_sparse_to_sparse_tensor(A_g))

    # Keras inputs
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_group_graphs = [
        tf.keras.layers.Input((n_nodes,), sparse=True)
        for _ in groups
    ]

    # Optimal k from the original adjacency
    k_opt = findOptimalK(adjacency)
    #k_opt = 20
    # Build model with MultiFairnessDMoN (multi-group modularity-gap loss)
    model = build_fairness_group_dmon(
        input_features,
        input_graph,
        input_adjacency,
        input_group_graphs,
        k_opt,
        lamda,
    )

    def grad(model, inputs):
        with tf.GradientTape() as tape:
            _ = model(inputs, training=True)
            loss_value = sum(model.losses)
        return model.losses, tape.gradient(loss_value, model.trainable_variables)

    lr_schedule = 0.01
    if n_nodes > 10000:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.07,
            decay_steps=100,
            decay_rate=0.9
        )
    '''if n_nodes <= 4000:
        lr_schedule = 0.001'''
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer, None)

    features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
    train_inputs = [features_tf, graph_normalized, graph] + group_sparse_graphs

    for epoch in range(1000):
        loss_values, grads = grad(model, train_inputs)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 200 == 0:
            print(
                f'epoch {epoch}, losses: ' +
                ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values])
            )

    # Get hard cluster assignments
    _, assignments = model(train_inputs, training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)

    communities_dict = {}
    for i, c in enumerate(clusters):
        communities_dict.setdefault(c, []).append(i)
    communities = list(communities_dict.values())

    return communities
