# coding=utf-8
# Multi-group extension of DMoN: max–min group modularity

import tensorflow.compat.v2 as tf


class MultiGroupDMoN(tf.keras.layers.Layer):
  """Multi-group Deep Modularity Network (DMoN) layer.

  This layer extends the original DMoN / groupDMoN to the multi-group setting.
  It implements the Max-Min Group Modularity objective described in the
  multi-group section:

      L = -Q(S) - λ_min^grp * Q_min^grp(S) + γ * R_collapse,

  where:
    - Q(S) is the standard modularity of the partition S,
    - Q_g(S) is the group modularity for group g (using A_g^{grp}),
    - Q_min^grp(S) = min_g Q_g(S),
    - R_collapse is the standard DMoN collapse regularizer.

  Inputs (in the order expected by deepClustering.py):
    inputs = [features, adjacency, A_g1, A_g2, ..., A_gm]
      features      : dense (n × d) tensor of node embeddings,
      adjacency     : sparse (n × n) adjacency of the full graph,
      A_gi          : sparse (n × n) group-specific adjacencies A_g^{grp}.

  Call signature:
    call(inputs, lamda)
      lamda >= 0 controls the strength of the max–min group term.
  """

  def __init__(self,
               n_clusters,
               collapse_regularization=0.1,
               dropout_rate=0.0,
               do_unpooling=False):
    super(MultiGroupDMoN, self).__init__()
    self.n_clusters = n_clusters
    self.collapse_regularization = collapse_regularization
    self.dropout_rate = dropout_rate
    self.do_unpooling = do_unpooling

  def build(self, input_shape):
    # input_shape is a list; we only need the feature dimension
    self.transform = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            self.n_clusters,
            kernel_initializer='orthogonal',
            bias_initializer='zeros'),
        tf.keras.layers.Dropout(self.dropout_rate),
    ])
    super(MultiGroupDMoN, self).build(input_shape)

  def call(self, inputs, lamda):
    """Compute pooled features and soft assignments with multi-group loss.

    Args:
      inputs: list/tuple [features, adjacency, A_g1, A_g2, ..., A_gm]
      lamda:  non-negative scalar; weight on the max–min group modularity term.

    Returns:
      features_pooled: (k × d) (or n × d if do_unpooling=True)
      assignments:     (n × k) soft cluster assignments
    """
    # Unpack inputs
    features = inputs[0]
    adjacency = inputs[1]
    group_adjacencies = inputs[2:]  # list of A_g^{grp}, one per group

    # Soft assignments S (n × k)
    assignments = tf.nn.softmax(self.transform(features), axis=1)
    cluster_sizes = tf.math.reduce_sum(assignments, axis=0)  # (k,)
    assignments_pooling = assignments / cluster_sizes        # (n × k)

    # Global modularity degrees, edges
    modularity_degrees = tf.sparse.reduce_sum(adjacency, axis=0)  # (n,)
    modularity_number_of_edges = tf.math.reduce_sum(modularity_degrees)
    degrees = tf.reshape(modularity_degrees, (-1, 1))  # (n × 1)

    number_of_nodes = adjacency.shape[1]

    # --- Global modularity loss: -Q(S) (same as original DMoN) ---

    graph_pooled = tf.transpose(
        tf.sparse.sparse_dense_matmul(adjacency, assignments))
    graph_pooled = tf.matmul(graph_pooled, assignments)

    normalizer_left = tf.matmul(assignments, degrees, transpose_a=True)   # (k × 1)
    normalizer_right = tf.matmul(degrees, assignments, transpose_a=True)  # (1 × k)
    normalizer = tf.matmul(normalizer_left, normalizer_right) / 2.0 / modularity_number_of_edges

    spectral_loss = -tf.linalg.trace(graph_pooled - normalizer) / 2.0 / modularity_number_of_edges
    # spectral_loss ≈ -Q(S)
    self.add_loss(spectral_loss)

    # --- Max-min group modularity term: -Q_min^grp(S) ---

    if lamda != 0 and len(group_adjacencies) > 0:
      Q_g_list = []

      for g_adj in group_adjacencies:
        # Degrees and edge count for group adjacency (used for null model)
        g_degrees = tf.sparse.reduce_sum(g_adj, axis=0)  # (n,)
        g_degrees = tf.reshape(g_degrees, (-1, 1))       # (n × 1)
        g_num_edges = tf.math.reduce_sum(g_degrees)      # 2 * m_g

        # If a group has no edges, skip it (avoids division by zero)
        # Q_g(S) will effectively be treated as 0 for such groups.
        def compute_Q_g():
          g_graph_pooled = tf.transpose(
              tf.sparse.sparse_dense_matmul(g_adj, assignments))
          g_graph_pooled = tf.matmul(g_graph_pooled, assignments)

          g_norm_left = tf.matmul(assignments, g_degrees, transpose_a=True)   # (k × 1)
          g_norm_right = tf.matmul(g_degrees, assignments, transpose_a=True)  # (1 × k)
          g_normalizer = tf.matmul(g_norm_left, g_norm_right) / 2.0 / g_num_edges

          # Group spectral loss scaled with global m (as in groupDMoN/fairDMoN):
          g_spectral_loss = tf.linalg.trace(g_graph_pooled - g_normalizer) / 2.0 / modularity_number_of_edges
          Q_g = g_spectral_loss  # Q_g(S)
          return Q_g

        Q_g = tf.cond(
            tf.greater(g_num_edges, 0.0),
            true_fn=compute_Q_g,
            false_fn=lambda: tf.constant(0.0, dtype=modularity_number_of_edges.dtype)
        )
        Q_g_list.append(Q_g)

      if Q_g_list:
        Q_g_tensor = tf.stack(Q_g_list)        # (G,)
        Q_min_grp = tf.reduce_min(Q_g_tensor)  # Q_min^grp(S)
        fairness_loss = -Q_min_grp            # -Q_min^grp(S)
        #self.add_loss(lamda * fairness_loss)
        # ---- NEW: simple edge-based scaling, no per-group normalization ----
        # approximate average group edge count (sum over group adjacencies)
        g_edge_counts = [
          tf.sparse.reduce_sum(g_adj)    # ≈ 2 * m_g
              for g_adj in group_adjacencies
        ]
        g_edges_mean = tf.add_n(g_edge_counts) / len(g_edge_counts)
        edge_ratio = modularity_number_of_edges / (g_edges_mean + 1e-8)


        # Now fairness_loss_scaled has roughly the SAME SCALE as spectral_loss
        fairness_loss = edge_ratio * fairness_loss

        # So λ in [0,1] is a proper trade-off knob
        self.add_loss(lamda * fairness_loss)

    # --- Collapse regularization (same as original DMoN) ---

    collapse_loss = tf.norm(cluster_sizes) / number_of_nodes * tf.sqrt(
        float(self.n_clusters)) - 1.0
    self.add_loss(self.collapse_regularization * collapse_loss)

    # --- Pooled features / unpooling ---

    features_pooled = tf.matmul(assignments_pooling, features, transpose_a=True)
    features_pooled = tf.nn.selu(features_pooled)
    if self.do_unpooling:
      features_pooled = tf.matmul(assignments_pooling, features_pooled)

    return features_pooled, assignments
  
  

class MultiFairnessDMoN(tf.keras.layers.Layer):
  """Multi-group Deep Modularity Network (DMoN) layer.

  This layer extends the original DMoN / groupDMoN to the multi-group setting.
  It implements the Max-Min Group Modularity objective described in the
  multi-group section:

      L = -Q(S) - λ_min^grp * Q_min^grp(S) + γ * R_collapse,

  where:
    - Q(S) is the standard modularity of the partition S,
    - Q_g(S) is the group modularity for group g (using A_g^{grp}),
    - Q_min^grp(S) = min_g Q_g(S),
    - R_collapse is the standard DMoN collapse regularizer.

  Inputs (in the order expected by deepClustering.py):
    inputs = [features, adjacency, A_g1, A_g2, ..., A_gm]
      features      : dense (n × d) tensor of node embeddings,
      adjacency     : sparse (n × n) adjacency of the full graph,
      A_gi          : sparse (n × n) group-specific adjacencies A_g^{grp}.

  Call signature:
    call(inputs, lamda)
      lamda >= 0 controls the strength of the max–min group term.
  """

  def __init__(self,
               n_clusters,
               collapse_regularization=0.1,
               dropout_rate=0.0,
               do_unpooling=False):
    super(MultiFairnessDMoN, self).__init__()
    self.n_clusters = n_clusters
    self.collapse_regularization = collapse_regularization
    self.dropout_rate = dropout_rate
    self.do_unpooling = do_unpooling

  def build(self, input_shape):
    # input_shape is a list; we only need the feature dimension
    self.transform = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            self.n_clusters,
            kernel_initializer='orthogonal',
            bias_initializer='zeros'),
        tf.keras.layers.Dropout(self.dropout_rate),
    ])
    super(MultiFairnessDMoN, self).build(input_shape)

  def call(self, inputs, lamda):
    """Compute pooled features and soft assignments with multi-group loss.

    Args:
      inputs: list/tuple [features, adjacency, A_g1, A_g2, ..., A_gm]
      lamda:  non-negative scalar; weight on the max–min group modularity term.

    Returns:
      features_pooled: (k × d) (or n × d if do_unpooling=True)
      assignments:     (n × k) soft cluster assignments
    """
    # Unpack inputs
    features = inputs[0]
    adjacency = inputs[1]
    group_adjacencies = inputs[2:]  # list of A_g^{grp}, one per group

    # Soft assignments S (n × k)
    assignments = tf.nn.softmax(self.transform(features), axis=1)
    cluster_sizes = tf.math.reduce_sum(assignments, axis=0)  # (k,)
    assignments_pooling = assignments / cluster_sizes        # (n × k)

    # Global modularity degrees, edges
    modularity_degrees = tf.sparse.reduce_sum(adjacency, axis=0)  # (n,)
    modularity_number_of_edges = tf.math.reduce_sum(modularity_degrees)
    degrees = tf.reshape(modularity_degrees, (-1, 1))  # (n × 1)

    number_of_nodes = adjacency.shape[1]

    # --- Global modularity loss: -Q(S) (same as original DMoN) ---

    graph_pooled = tf.transpose(
        tf.sparse.sparse_dense_matmul(adjacency, assignments))
    graph_pooled = tf.matmul(graph_pooled, assignments)

    normalizer_left = tf.matmul(assignments, degrees, transpose_a=True)   # (k × 1)
    normalizer_right = tf.matmul(degrees, assignments, transpose_a=True)  # (1 × k)
    normalizer = tf.matmul(normalizer_left, normalizer_right) / 2.0 / modularity_number_of_edges

    spectral_loss = -tf.linalg.trace(graph_pooled - normalizer) / 2.0 / modularity_number_of_edges
    # spectral_loss ≈ -Q(S)
    self.add_loss(spectral_loss)

    # --- Max-min group modularity term: -Q_min^grp(S) ---

    if lamda != 0 and len(group_adjacencies) > 0:
      Q_g_list = []

      for g_adj in group_adjacencies:
        # Degrees and edge count for group adjacency (used for null model)
        g_degrees = tf.sparse.reduce_sum(g_adj, axis=0)  # (n,)
        g_degrees = tf.reshape(g_degrees, (-1, 1))       # (n × 1)
        g_num_edges = tf.math.reduce_sum(g_degrees)      # 2 * m_g

        # If a group has no edges, skip it (avoids division by zero)
        # Q_g(S) will effectively be treated as 0 for such groups.
        def compute_Q_g():
          g_graph_pooled = tf.transpose(
              tf.sparse.sparse_dense_matmul(g_adj, assignments))
          g_graph_pooled = tf.matmul(g_graph_pooled, assignments)

          g_norm_left = tf.matmul(assignments, g_degrees, transpose_a=True)   # (k × 1)
          g_norm_right = tf.matmul(g_degrees, assignments, transpose_a=True)  # (1 × k)
          g_normalizer = tf.matmul(g_norm_left, g_norm_right) / 2.0 / g_num_edges

          # Group spectral loss scaled with global m (as in groupDMoN/fairDMoN):
          g_spectral_loss = tf.linalg.trace(g_graph_pooled - g_normalizer) / 2.0 / modularity_number_of_edges
          Q_g = g_spectral_loss  # Q_g(S)
          return Q_g

        Q_g = tf.cond(
            tf.greater(g_num_edges, 0.0),
            true_fn=compute_Q_g,
            false_fn=lambda: tf.constant(0.0, dtype=modularity_number_of_edges.dtype)
        )
        Q_g_list.append(Q_g)

      if Q_g_list:
        Q_g_tensor = tf.stack(Q_g_list)        # (G,)
        Q_min_grp = tf.reduce_min(Q_g_tensor)  # Q_min^grp(S)
        Q_max_grp = tf.reduce_max(Q_g_tensor)
        fairness_loss = tf.abs(Q_max_grp - Q_min_grp)            # -Q_min^grp(S)
        self.add_loss(lamda * fairness_loss)
        

    # --- Collapse regularization (same as original DMoN) ---

    collapse_loss = tf.norm(cluster_sizes) / number_of_nodes * tf.sqrt(
        float(self.n_clusters)) - 1.0
    
    self.add_loss(self.collapse_regularization * collapse_loss)

    # --- Pooled features / unpooling ---

    features_pooled = tf.matmul(assignments_pooling, features, transpose_a=True)
    features_pooled = tf.nn.selu(features_pooled)
    if self.do_unpooling:
      features_pooled = tf.matmul(assignments_pooling, features_pooled)

    return features_pooled, assignments

