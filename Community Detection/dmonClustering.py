
import sys
import os
import csv



import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator,eigsh

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
tf.compat.v1.enable_v2_behavior()



def load_graphGroup(edgelist_file, features_file, featuresType,redAttr):
    # Load edges
    edges = pd.read_csv(edgelist_file, sep=' ', header=None)
    nodes = pd.concat([edges[0], edges[1]]).unique()
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    row = edges[0].map(node_to_idx)
    col = edges[1].map(node_to_idx)
    adjacency = sp.coo_matrix((np.ones(len(edges)), (row, col)),
                              shape=(len(nodes), len(nodes))).tocsr()

    # Make symmetric undirected
    adjacency = (adjacency + adjacency.T).minimum(1)

    # Load node attributes
    features_data = pd.read_csv(features_file)
    features_data['node_idx'] = features_data['nodes'].map(node_to_idx)
    node_attribute_dict = dict(zip(features_data['node_idx'], features_data['attribute']))
    node_attribute_df = pd.DataFrame({'node_idx': features_data['node_idx'],
                                      'attribute': features_data['attribute']})

    # Build red/blue sparse adjacency directly from nonzero entries
    coo = adjacency.tocoo()
    mask_red  = [(node_attribute_dict.get(i)==redAttr or node_attribute_dict.get(j)==redAttr)
                 for i,j in zip(coo.row, coo.col)]
    mask_blue = [(node_attribute_dict.get(i)==1-redAttr or node_attribute_dict.get(j)==1-redAttr)
                 for i,j in zip(coo.row, coo.col)]

    red_adjacency  = sp.coo_matrix((coo.data[mask_red], (coo.row[mask_red], coo.col[mask_red])),
                                   shape=adjacency.shape).tocsr()
    blue_adjacency = sp.coo_matrix((coo.data[mask_blue], (coo.row[mask_blue], coo.col[mask_blue])),
                                   shape=adjacency.shape).tocsr()

    # Build feature matrix
    feature_rows = features_data['node_idx'].astype(int).to_numpy()

    if featuresType == 'degree':
        degrees = defaultdict(int)
        for u, v in zip(coo.row, coo.col):
            degrees[u] += 1
            degrees[v] += 1
        values = np.array([degrees[i] for i in feature_rows])
        features = sp.csr_matrix((values, (feature_rows, feature_rows)), shape=(len(nodes), len(nodes)))
    elif featuresType == 'id':
        features = sp.csr_matrix((feature_rows, (feature_rows, feature_rows)), shape=(len(nodes), len(nodes)))
    else:
        attributes = features_data['attribute'].to_numpy()
        features = sp.csr_matrix((attributes, (feature_rows, np.zeros_like(feature_rows))),
                                 shape=(len(nodes), 1))

    return adjacency, red_adjacency, blue_adjacency, features, node_attribute_dict, node_attribute_df


def load_graphDiversity(edgelist_file, features_file, featuresType):
    # Load edges
    edges = pd.read_csv(edgelist_file, sep=' ', header=None)
    nodes = pd.concat([edges[0], edges[1]]).unique()
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    row = edges[0].map(node_to_idx)
    col = edges[1].map(node_to_idx)
    adjacency = sp.coo_matrix((np.ones(len(edges)), (row, col)),
                            shape=(len(nodes), len(nodes))).tocsr()

    # Make symmetric undirected
    adjacency = (adjacency + adjacency.T).minimum(1)

    # Load node attributes
    features_data = pd.read_csv(features_file)
    features_data['node_idx'] = features_data['nodes'].map(node_to_idx)
    node_attribute_dict = dict(zip(features_data['node_idx'], features_data['attribute']))
    node_attribute_df = pd.DataFrame({'node_idx': features_data['node_idx'],
                                    'attribute': features_data['attribute']})

    # Build red/blue sparse adjacency directly from nonzero entries
    coo = adjacency.tocoo()
    mask_div  = [(node_attribute_dict.get(i) != node_attribute_dict.get(j))
                for i,j in zip(coo.row, coo.col)]
    
    diversity_adjacency  = sp.coo_matrix((coo.data[mask_div], (coo.row[mask_div], coo.col[mask_div])),
                                shape=adjacency.shape).tocsr()
    
    # Build feature matrix
    feature_rows = features_data['node_idx'].astype(int).to_numpy()
    print('creating feature matrix\n')
    if featuresType == 'degree':
        degrees = defaultdict(int)
        for u, v in zip(coo.row, coo.col):
            degrees[u] += 1
            degrees[v] += 1
        values = np.array([degrees[i] for i in feature_rows])
        features = sp.csr_matrix((values, (feature_rows, feature_rows)), shape=(len(nodes), len(nodes)))
    elif featuresType == 'id':
        features = sp.csr_matrix((feature_rows, (feature_rows, feature_rows)), shape=(len(nodes), len(nodes)))
    else:
        attributes = features_data['attribute'].to_numpy()
        features = sp.csr_matrix((attributes, (feature_rows, np.zeros_like(feature_rows))),
                                shape=(len(nodes), 1))

    return adjacency,diversity_adjacency, features,node_attribute_dict,node_attribute_df



def findOptimalK(A, max_eigs=100):
    n = A.shape[0]
    d = np.array(A.sum(axis=1)).flatten().astype(float)
    m = d.sum()/2

    sqrt_d = np.sqrt(d, where=(d>0), out=np.zeros_like(d))
    inv_sqrt = np.divide(1.0, sqrt_d, where=(sqrt_d>0), out=np.zeros_like(d))

    def matvec(x):
        y = inv_sqrt * x
        first = inv_sqrt * (A.dot(y))
        correction = (d.dot(y)/(2*m)) * inv_sqrt * d
        return first - correction

    B_op = LinearOperator((n,n), matvec=matvec, dtype=float)

    # Only compute top max_eigs eigenvalues — far cheaper than 100
    eigvals = eigsh(B_op, k=min(max_eigs, n-2), which="LA", return_eigenvectors=False)
    sorted_vals = np.sort(eigvals)[::-1]
    gaps = np.diff(sorted_vals)
    # Pick second-largest gap
    idxs = np.argsort(gaps)
    return idxs[-1] + 1

def convert_scipy_sparse_to_sparse_tensor(
    matrix):
  """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

  Args:
    matrix: A scipy sparse matrix.

  Returns:
    A ternsorflow sparse matrix (rank-2 tensor).
  """
  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)
  

from tools import dmon

#from graph_embedding.dmon.dmon_edgePerc_loss import DMoNredPerc   # Ensure this imports your modified DMoN class

def build_dmon(input_features,
               input_graph,
               input_adjacency, clustersNumber):

  output = input_features

  num_nodes = input_graph.shape[1]

  
  for n_channels in [64]: #defines the number of channels (neurons) in each GCN layer
    output = gcn.GCN(num_nodes,n_channels)([output, input_graph])


  pool, pool_assignment = dmon.DMoN(
      clustersNumber,
      collapse_regularization=1,
      dropout_rate=0.2)([output, input_adjacency])

  return tf.keras.Model(
      inputs=[input_features, input_graph, input_adjacency],
      outputs=[pool, pool_assignment])


def diversityDMoNClustering(edgelist_path,attributes_path,featuresType = 'id',lamda = 0.5):
    
    adjacency,diversity_adjacency, features,node_attributes_dict,node_attribute_df = load_graphDiversity(edgelist_path,attributes_path,featuresType)
    modified_adjacency = ((1-lamda)*adjacency +  lamda* diversity_adjacency).tocsr()
    
    original_features =features.copy()
    diag = original_features.diagonal().astype(np.float32)
    features = diag.reshape(-1, 1)

    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]
    graph_modified = convert_scipy_sparse_to_sparse_tensor(modified_adjacency)
    graph_normalized_modified = convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(modified_adjacency.copy()))
    


    # Create model input placeholders of appropriate size
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_red_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_red_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    
    k_opt = findOptimalK(adjacency)
    
    model = build_dmon(input_features, input_red_graph, input_red_adjacency,k_opt)
    
    # Computes the gradients wrt. the sum of losses, returns a list of them.
    def grad(model, inputs):
        with tf.GradientTape() as tape:
                _ = model(inputs, training=True)
                loss_value = sum(model.losses)
        return model.losses, tape.gradient(loss_value, model.trainable_variables)
    lr_schedule = 0.01
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer, None)
    
    
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    for epoch in range(1000):

        
        loss_values, grads = grad(model, [features, graph_normalized_modified, graph_modified])
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 100 == 0:
            print(f'epoch {epoch}, losses: ' +
                    ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))
            
    _, assignments = model([features, graph_normalized_modified, graph_modified], training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
    #print('number of nodes:',adjacency.shape[0])
    #print('clusters:', clusters)
    communities_dict = {}
    for i in range(len(clusters)):
        if clusters[i] in communities_dict:
            communities_dict[clusters[i]].append(i)
        else:
            communities_dict[clusters[i]] = [i]
    communities = list(communities_dict.values())
     
    return communities



def groupDMoNClustering(edgelist_path,attributes_path,featuresType = 'id',lamda = 0.5):
    
    redAttr = 1
    
    adjacency,red_adjacency, blue_adjacency, features,node_attributes_dict,node_attribute_df = load_graphGroup(edgelist_path,attributes_path,featuresType,redAttr)
    modified_adjacency = ((1-lamda)*adjacency +  lamda* red_adjacency).tocsr()
    
    original_features =features.copy()
    diag = original_features.diagonal().astype(np.float32)
    features = diag.reshape(-1, 1)

    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]
    graph_modified = convert_scipy_sparse_to_sparse_tensor(modified_adjacency)
    graph_normalized_modified = convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(modified_adjacency.copy()))
    


    # Create model input placeholders of appropriate size
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_red_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_red_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    
    k_opt = findOptimalK(adjacency)
    
    model = build_dmon(input_features, input_red_graph, input_red_adjacency,k_opt)
    
    # Computes the gradients wrt. the sum of losses, returns a list of them.
    def grad(model, inputs):
        with tf.GradientTape() as tape:
                _ = model(inputs, training=True)
                loss_value = sum(model.losses)
        return model.losses, tape.gradient(loss_value, model.trainable_variables)
    lr_schedule = 0.01
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer, None)
    
    
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    for epoch in range(1000):

        
        loss_values, grads = grad(model, [features, graph_normalized_modified, graph_modified])
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 100 == 0:
            print(f'epoch {epoch}, losses: ' +
                    ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))
            
    _, assignments = model([features, graph_normalized_modified, graph_modified], training=False)
    assignments = assignments.numpy()
    clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
    #print('number of nodes:',adjacency.shape[0])
    #print('clusters:', clusters)
    communities_dict = {}
    for i in range(len(clusters)):
        if clusters[i] in communities_dict:
            communities_dict[clusters[i]].append(i)
        else:
            communities_dict[clusters[i]] = [i]
    communities = list(communities_dict.values())
     
    return communities