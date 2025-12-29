
import sys
import os

from scipy.sparse import diags


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator,eigsh





import scipy.sparse as sp
from sklearn.cluster import KMeans
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition

import tensorflow.compat.v2 as tf





def compute_ModMatrix_lamda(x,A, d,m, A_blue, dB, mB, lamda=0.5):

    sqrt_dB = np.sqrt(dB, where=(dB>0), out=np.zeros_like(dB))
    inv_sqrt_dB = np.divide(1.0, sqrt_dB, where=(sqrt_dB>0), out=np.zeros_like(dB))

    y = inv_sqrt_dB * x
    first_termBlueMod = inv_sqrt_dB * (A_blue.dot(y))

    dTy = dB.dot(y)
    correctionBlueMod = (dTy / (2*mB)) * inv_sqrt_dB * dB

    sqrt_d = np.sqrt(d, where=(d>0), out=np.zeros_like(d))
    inv_sqrt_d = np.divide(1.0, sqrt_d, where=(sqrt_d>0), out=np.zeros_like(d))

    y = inv_sqrt_d * x
    first_termMod = inv_sqrt_d * (A.dot(y))

    dTy = dB.dot(y)
    correctionMod = (dTy / (2*m)) * inv_sqrt_d * d

    return   ((1-lamda) * first_termMod - correctionMod)+ (lamda *(first_termBlueMod - correctionBlueMod))


def spectralDiversityClustering(adjacency,node_attributes_dict,lamda=0.5):
    
    k_trial = 100
    

    G = nx.Graph(adjacency)
    nx.set_node_attributes(G, node_attributes_dict, 'attribute')

    A = adjacency.tocsr()
    # Print the number of nodes and edges in the graph

    nodelist = list(G)
    nlen = len(nodelist)

    indexNodes = dict(zip(range(nlen),nodelist))
     

    # Create the adjacency matrix of G
    A = nx.adjacency_matrix(G)
    n = len(G.nodes())
    
    # Create a random adjacency matrix A of n number of nodes
    A = csr_matrix(A)
    
    # Degree vector (sum of rows of A)
    d = np.array(A.sum(axis=1)).flatten()

    # Total number of edges in the graph (half the sum of all degrees)
    m = d.sum() / 2
        
    # Dimension of the matrix
    n = A.shape[0]
     
    A_coo = A.tocoo()
    attr = np.array([node_attributes_dict[indexNodes[i]] for i in range(A.shape[0])])
    keep = (attr[A_coo.row] != attr[A_coo.col])

    
    A_div = csr_matrix(
        (np.ones(keep.sum()), (A_coo.row[keep], A_coo.col[keep])),
        shape=A.shape
    )
    
    
    ddiv = np.array(A_div.sum(axis=1)).flatten()

    # Total edge count = sum of degrees / 2 (undirected)
    mB = ddiv.sum() / 2

    #find k_opt
    
    B = LinearOperator((n, n), matvec=lambda x: compute_ModMatrix_lamda(x,A, d,m, A_div, ddiv, mB.flatten(), 0))
    eigenvalues, eigenvectors = eigs(B, k=k_trial, which='LR', v0=np.random.rand(n),ncv=4*k_trial,maxiter=200000)
    eigs_real = eigenvalues.real + 1  # +1 because gap i is between eig[i] and eig[i+1]
    
    eigs_sorted = np.sort(eigs_real)[::-1]
    
    gaps = np.diff(eigs_sorted)
    k_opt = np.argmax(gaps) + 1  # +1 because gap i is between eig[i] and eig[i+1]
    
    BB = LinearOperator((n, n), matvec=lambda x: compute_ModMatrix_lamda(x,A, d,m, A_div, ddiv, mB.flatten(), lamda))

    eigenvalues, eigenvectors = eigs(BB, k=k_opt, which='LR', v0=np.random.rand(n),ncv=4*k_trial,maxiter=200000)



    eigs_real = eigenvalues.real + 1  # +1 because gap i is between eig[i] and eig[i+1]
    
    V = eigenvectors[:, np.argsort(eigenvalues.real)[-k_opt:]] * np.sqrt(eigenvalues.real[-k_opt:])
    idx = np.argsort(eigenvalues.real)[-k_opt:]
    V = eigenvectors[:, idx].real
    kmeans = KMeans(n_clusters=k_opt, init='k-means++', random_state=0).fit(V)
    
    labels = kmeans.labels_  # This assigns a community label to each node
    community_dict= {}

    for id, community_id in enumerate(labels):
        #print(id, community_id)
        node_id = indexNodes[id]
        if community_id in community_dict:
            community_dict[community_id].append(node_id)
        else:
            community_dict[community_id] = [node_id]
    communities = community_dict.values()
    
    return communities


def spectralGroupClustering(adjacency,node_attributes_dict,lamda=0.5):
    
    k_trial = 100
    

    G = nx.Graph(adjacency)
    nx.set_node_attributes(G, node_attributes_dict, 'attribute')

    A = adjacency.tocsr()
    # Print the number of nodes and edges in the graph

    nodelist = list(G)
    nlen = len(nodelist)

    indexNodes = dict(zip(range(nlen),nodelist))
     

    # Create the adjacency matrix of G
    A = nx.adjacency_matrix(G)
    n = len(G.nodes())
    
    # Create a random adjacency matrix A of n number of nodes
    A = csr_matrix(A)
    
    # Degree vector (sum of rows of A)
    d = np.array(A.sum(axis=1)).flatten()

    # Total number of edges in the graph (half the sum of all degrees)
    m = d.sum() / 2
        
    # Dimension of the matrix
    n = A.shape[0]
     
    A_coo = A.tocoo()
    attr = np.array([node_attributes_dict[indexNodes[i]] for i in range(A.shape[0])])
    keep = (attr[A_coo.row] == 0) | (attr[A_coo.col] == 0)

    A_blue = csr_matrix(
        (np.ones(keep.sum()), (A_coo.row[keep], A_coo.col[keep])),
        shape=A.shape
    )
    
    
    
    dB = np.array(A_blue.sum(axis=1)).flatten()

    # Total edge count = sum of degrees / 2 (undirected)
    mB = dB.sum() / 2

    #find k_opt
    
    B = LinearOperator((n, n), matvec=lambda x: compute_ModMatrix_lamda(x,A, d,m, A_blue, dB, mB.flatten(), 0))
    eigenvalues, eigenvectors = eigs(B, k=k_trial, which='LR', v0=np.random.rand(n),ncv=4*k_trial,maxiter=200000)
    eigs_real = eigenvalues.real + 1  # +1 because gap i is between eig[i] and eig[i+1]
    
    eigs_sorted = np.sort(eigs_real)[::-1]
    
    gaps = np.diff(eigs_sorted)
    k_opt = np.argmax(gaps) + 1  # +1 because gap i is between eig[i] and eig[i+1]
    
    BB = LinearOperator((n, n), matvec=lambda x: compute_ModMatrix_lamda(x,A, d,m, A_blue, dB, mB.flatten(), lamda))

    eigenvalues, eigenvectors = eigs(BB, k=k_opt, which='LR', v0=np.random.rand(n),ncv=4*k_trial,maxiter=200000)



    eigs_real = eigenvalues.real + 1  # +1 because gap i is between eig[i] and eig[i+1]
    
    V = eigenvectors[:, np.argsort(eigenvalues.real)[-k_opt:]] * np.sqrt(eigenvalues.real[-k_opt:])
    idx = np.argsort(eigenvalues.real)[-k_opt:]
    V = eigenvectors[:, idx].real
    kmeans = KMeans(n_clusters=k_opt, init='k-means++', random_state=0).fit(V)
    
    labels = kmeans.labels_  # This assigns a community label to each node
    community_dict= {}

    for id, community_id in enumerate(labels):
        #print(id, community_id)
        node_id = indexNodes[id]
        if community_id in community_dict:
            community_dict[community_id].append(node_id)
        else:
            community_dict[community_id] = [node_id]
    communities = community_dict.values()
    
    return communities