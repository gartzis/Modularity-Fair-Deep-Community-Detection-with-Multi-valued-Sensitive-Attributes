import sys
import os

from networkx.algorithms.community import louvain_communities


sys.path.append('Algorithms')

from diversityFairness import diversityMetric
from modularityFairness import modularityFairnessMetric as multiModularityFairnessMetric
from L_diversityFairness import LdiversityMetric as LDiversityFairnessMetric
from L_modularityFairness import LModularityFairnessMetric


sys.path.append('Community Detection')
from multi_deepClustering import  deepMultiGroupClustering, deepMultiGroupDiversityClustering, deepMultiFairnessClustering
from deepClustering import deepDiversityClustering

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, LinearOperator, eigsh

from sklearn.cluster import KMeans
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition



def load_graph_from_files(edgelist_file, features_file):
    """
    Load graph from .edgelist and node attributes from .csv.
    Returns:
        adjacency: scipy.sparse adjacency matrix (CSR)
        node_attribute_dict: {node_idx: attribute_value}
        node_attribute_df: DataFrame with columns ['node_idx', 'attribute']
    """
    edges = pd.read_csv(edgelist_file, sep=' ', header=None)
    nodes = pd.concat([edges[0], edges[1]]).unique()
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    row = edges[0].map(node_to_idx)
    col = edges[1].map(node_to_idx)
    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (row, col)),
        shape=(len(nodes), len(nodes))
    ).tocsr()

    # Make undirected and unweighted (0/1)
    adjacency = (adjacency + adjacency.T).minimum(1)

    features_data = pd.read_csv(features_file)
    features_data['node_idx'] = features_data['nodes'].map(node_to_idx)
    node_attribute_dict = dict(
        zip(features_data['node_idx'], features_data['attribute'])
    )
    node_attribute_df = pd.DataFrame(
        {'node_idx': features_data['node_idx'],
         'attribute': features_data['attribute']}
    )

    return adjacency, node_attribute_dict, node_attribute_df



class NotAPartition(NetworkXError):
    """Raised if a given collection is not a partition."""
    def __init__(self, G, collection):
        msg = f"{collection} is not a valid partition of the graph {G}"
        super().__init__(msg)


def modularityCustom(G, communities, weight="weight", resolution=1):
    """
    Copy of your custom modularity to also get per-community modularity list.
    """
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)

    directed = G.is_directed()
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        m = sum(out_degree.values())
        norm = 1 / m**2
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum**2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(
            wt
            for u, v, wt in G.edges(comm, data=weight, default=1)
            if v in comm
        )

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = (sum(in_degree[u] for u in comm)
                         if directed else out_degree_sum)

        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm

    communityModularityList = []
    for community in communities:
        communityModularityList.append(community_contribution(community))

    return sum(map(community_contribution, communities)), communityModularityList


def computeMetrics(G, communities, G_attribute):
    """
    Compute structural modularity + multi-group fairness/diversity metrics.

    Uses:
      - diversityMetric (multi-group diversity)
      - multiModularityFairnessMetric (multi-group group-modularity fairness)
      - LModularityFairnessMetric (multi-group L-modularity fairness)
      - LDiversityFairnessMetric (multi-group L-diversity)
    """
    # Standard structural modularity (no attributes)
    modularity = nx.algorithms.community.modularity(
        G, communities, weight="weight"
    )

    # Multi-group diversity (cross-group) modularity
    diversity_modularity, diversityModularityList = diversityMetric(
        G, communities, G_attribute, weight="weight", resolution=1
    )

    # Multi-group group-modularity fairness
    (
        unfairness_gap,
        unfairness_per_community,
        unfairness_normalized,
        per_group_Q_list,
        per_group_Q,
    ) = multiModularityFairnessMetric(
        G, communities, G_attribute, weight="weight", resolution=1
    )

    # Multi-group L-modularity fairness
    (
        lUnfairness_gap,
        lUnfairness_per_community,
        lUnfairness_normalized,
        per_group_L_list,
        per_group_L,
    ) = LModularityFairnessMetric(
        G, communities, G_attribute, weight="weight", resolution=1
    )

    # Multi-group L-diversity
    lDiversity, lDiversityList = LDiversityFairnessMetric(
        G, communities, G_attribute, weight="weight", resolution=1
    )

    groups = sorted(set(G_attribute.values()))

    print('\nModularity (structural):', modularity)
    print('---------------------')
    print('Group modularities Q_g (multi-group):')
    for g in groups:
        print(f'  Group {g}: Q_g = {per_group_Q[g]:.4f}')
    print('L-Group modularities Q_g^L (multi-group):')
    for g in groups:
        print(f'  Group {g}: Q_g^L = {per_group_L[g]:.4f}')
    print('---------------------')
    print('Unfairness gap (max_g Q_g - min_g Q_g):', unfairness_gap)
    print('Normalized unfairness:', 1- unfairness_normalized)
    print('L-Unfairness gap (L version):', lUnfairness_gap)
    print('L-Normalized unfairness:', lUnfairness_normalized)
    print('---------------------')
    print('Diversity (cross-group modularity):', diversity_modularity)
    print('L-Diversity:', lDiversity)

    return




def plotCommunitiesMulti(G, node_attributes_dict, communities,
                         file_name, plotName, method):
    """
    Multi-group version of your plotting routine.

    - Each community has its own marker shape.
    - Each attribute value (group) has its own color.
    """

    # Marker shapes for different communities
    community_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', 'X']
    # Color palette for attribute groups
    # (tab10 gives up to 10 distinct colors; extend if needed)
    cmap = plt.get_cmap('tab10')

    plt.figure().set_size_inches(22, 19)
    pos = nx.spring_layout(G)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Determine all attribute values present in these communities
    all_nodes = [node for community in communities for node in community]
    all_attrs = sorted({node_attributes_dict[n] for n in all_nodes})
    group_to_color = {
        g: cmap(i % 10) for i, g in enumerate(all_attrs)
    }

    # Draw nodes per community, colored by attribute
    for community_id, community_nodes in enumerate(communities):
        community_marker = community_markers[community_id % len(community_markers)]

        node_colors = [
            group_to_color[node_attributes_dict[node]]
            for node in community_nodes
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=community_nodes,
            node_color=node_colors,
            label=f'Community {community_id}',
            node_size=300,
            alpha=1,
            linewidths=1,
            node_shape=community_marker,
        )

    nx.draw_networkx_labels(G, pos, font_size=10)

    # Legend: one entry per community (marker) + one entry per group (color)
    legend_patches = []
    for community_id in range(len(communities)):
        community_marker = community_markers[community_id % len(community_markers)]
        patch = plt.Line2D(
            [0], [0],
            marker=community_marker,
            color='w',
            label=f'Community {community_id}',
            markerfacecolor='gray',
            markersize=10
        )
        legend_patches.append(patch)

    # Group legend (by attribute value)
    group_patches = []
    for g in all_attrs:
        patch = plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f'Group {g}',
            markerfacecolor=group_to_color[g],
            markersize=10
        )
        group_patches.append(patch)

    plt.legend(handles=legend_patches + group_patches, loc='best')

    # Build output directory
    out_dir = os.path.join('Synth Results', method, file_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{plotName}.png")
    plt.savefig(out_path)
    plt.close()



asymetric_data_path = 'Data//Multi-Group'

file_patterns = ['Asymmetric_1000_5_K4_025_025_025_025_09_09_09_09_01_05_05_05_9' ]

asymetric_file_paths = [os.path.join(asymetric_data_path, file) for file in os.listdir(asymetric_data_path) if 'csv' not in file and any(pattern in file for pattern in file_patterns)]


datasets = []

for file_path in asymetric_file_paths:

    for file in os.listdir(file_path):

        file = file.split('.')[0]
        if file not in datasets and 'Backup' not in file and 'backup' not in file and 'Original' not in file and 'original' not in file:
            datasets.append(file)







if not os.path.exists('Synth Results'):
    os.makedirs('Synth Results')
# Ensure root results dir exists
if not os.path.exists('Synth Results'):
    os.makedirs('Synth Results')


for file_path in datasets:
    datasetName = file_path.split('.')[0]
    datasetName = file_path.split('//')[-1].split('.')[0]

    #print(datasetName)
    if "Multi-Group" in asymetric_data_path:
        base_dir = 'Data//Multi-Group//' + file_path.split('.csv')[0] + '\\'
    print(datasetName)  
    datasetRead = base_dir + '\\' + datasetName

    # Load graph + attributes
    adjacency, graph_attributes, node_attribute_df = load_graph_from_files(
        datasetRead + '.edgelist',
        datasetRead + '.csv'
    )
    
    

    graph = nx.Graph(adjacency)
    print('Nodes:',len(graph.nodes))
    print('Edges:', len(graph.edges))
    print('Different Attributes:', len(set(graph_attributes.values())))
    nx.set_node_attributes(graph, graph_attributes, 'attribute')
    


    print('\n--- Multi-Group Deep Group ---')
    method_name = 'Multi Deep Group Communities'

    # Create result directories
    if not os.path.exists(os.path.join('Synth Results', method_name)):
        os.makedirs(os.path.join('Synth Results', method_name))
    if not os.path.exists(os.path.join('Synth Results', method_name, datasetName)):
        os.makedirs(os.path.join('Synth Results', method_name, datasetName))

    # Run multi-group DeepGroup
    communities = deepMultiGroupClustering(
        datasetRead + '.edgelist',
        datasetRead + '.csv',
        lamda=1
    )

    print('Number of communities:', len(communities))

    # Plot
    plotName = datasetName + '_MultiDeepGroup'
    plotCommunitiesMulti(
        graph,
        graph_attributes,
        communities,
        datasetName,
        plotName,
        method_name
    )

    # Save communities to CSV
    community_df = pd.DataFrame(
        [(node, community)
         for community, nodes in enumerate(communities)
         for node in nodes],
        columns=['nodes', 'community']
    )

    out_csv = os.path.join('Synth Results', method_name, datasetName,
                           datasetName + '_communities.csv')
    community_df.to_csv(out_csv, index=False)

    # Compute metrics (binary-focused for now)
    computeMetrics(graph, communities, graph_attributes.copy())


    print('\n--- Multi-Diversity Deep Group ---')
    method_name = 'Multi Deep Diversity Communities'

    # Create result directories
    if not os.path.exists(os.path.join('Synth Results', method_name)):
        os.makedirs(os.path.join('Synth Results', method_name))
    if not os.path.exists(os.path.join('Synth Results', method_name, datasetName)):
        os.makedirs(os.path.join('Synth Results', method_name, datasetName))

    # Run multi-group DeepDiversity
    communities = deepMultiGroupDiversityClustering(
        datasetRead + '.edgelist',
        datasetRead + '.csv',
        lamda=1
    )

    print('Number of communities:', len(communities))

    # Plot
    plotName = datasetName + '_MultiDeepDiversity'
    plotCommunitiesMulti(
        graph,
        graph_attributes,
        communities,
        datasetName,
        plotName,
        method_name
    )

    # Save communities to CSV
    community_df = pd.DataFrame(
        [(node, community)
         for community, nodes in enumerate(communities)
         for node in nodes],
        columns=['nodes', 'community']
    )

    out_csv = os.path.join('Synth Results', method_name, datasetName,
                           datasetName + '_communities.csv')
    community_df.to_csv(out_csv, index=False)

    # Compute metrics (binary-focused for now)
    computeMetrics(graph, communities, graph_attributes.copy())


    print('\n--- Multi Deep Fairness ---')
    method_name = 'Multi Deep Fairness Communities'

    # Create result directories
    if not os.path.exists(os.path.join('Synth Results', method_name)):
        os.makedirs(os.path.join('Synth Results', method_name))
    if not os.path.exists(os.path.join('Synth Results', method_name, datasetName)):
        os.makedirs(os.path.join('Synth Results', method_name, datasetName))

    # Run multi-group DeepFairness
    communities = deepMultiFairnessClustering(
        datasetRead + '.edgelist',
        datasetRead + '.csv',
        lamda=1
    )

    print('Number of communities:', len(communities))

    # Plot
    plotName = datasetName + '_MultiDeepFairness'
    plotCommunitiesMulti(
        graph,
        graph_attributes,
        communities,
        datasetName,
        plotName,
        method_name
    )

    # Save communities to CSV
    community_df = pd.DataFrame(
        [(node, community)
         for community, nodes in enumerate(communities)
         for node in nodes],
        columns=['nodes', 'community']
    )

    out_csv = os.path.join('Synth Results', method_name, datasetName,
                           datasetName + '_communities.csv')
    community_df.to_csv(out_csv, index=False)

    # Compute metrics (binary-focused for now; you can later swap in multi-group metrics)
    computeMetrics(graph, communities, graph_attributes.copy())
    
    

    print('\n--- Deep  Diversity ---')
    method_name = 'Deep  Diversity Communities'

    # Create result directories
    if not os.path.exists(os.path.join('Synth Results', method_name)):
        os.makedirs(os.path.join('Synth Results', method_name))
    if not os.path.exists(os.path.join('Synth Results', method_name, datasetName)):
        os.makedirs(os.path.join('Synth Results', method_name, datasetName))

    # Run multi-group DeepFairness
    communities = deepDiversityClustering(
        datasetRead + '.edgelist',
        datasetRead + '.csv',
        lamda=1
    )

    print('Number of communities:', len(communities))

    # Plot
    plotName = datasetName + '_DeepDiversity'
    plotCommunitiesMulti(
        graph,
        graph_attributes,
        communities,
        datasetName,
        plotName,
        method_name
    )

    # Save communities to CSV
    community_df = pd.DataFrame(
        [(node, community)
         for community, nodes in enumerate(communities)
         for node in nodes],
        columns=['nodes', 'community']
    )

    out_csv = os.path.join('Synth Results', method_name, datasetName,
                           datasetName + '_communities.csv')
    community_df.to_csv(out_csv, index=False)

    # Compute metrics (binary-focused for now; you can later swap in multi-group metrics)
    computeMetrics(graph, communities, graph_attributes.copy())
    

    

    
    
    
    
    
    
