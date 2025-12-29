
import sys
import os

from networkx.algorithms.community import louvain_communities









sys.path.append('Algorithms')
from diversityFairness import diversityMetric
from modularityFairness import modularityFairnessMetric
from L_diversityFairness import LDiversityFairnessMetric
from L_modularityFairness import LModularityFairnessMetric


sys.path.append('Community Detection')



from dmonClustering import diversityDMoNClustering, groupDMoNClustering





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


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



def load_graph_from_files(edgelist_file, features_file):

    edges = pd.read_csv(edgelist_file, sep=' ', header=None)
    nodes = pd.concat([edges[0], edges[1]]).unique()
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    row = edges[0].map(node_to_idx)
    col = edges[1].map(node_to_idx)
    adjacency = sp.coo_matrix((np.ones(len(edges)), (row, col)),
                              shape=(len(nodes), len(nodes))).tocsr()


    adjacency = (adjacency + adjacency.T).minimum(1)


    features_data = pd.read_csv(features_file)
    features_data['node_idx'] = features_data['nodes'].map(node_to_idx)
    node_attribute_dict = dict(zip(features_data['node_idx'], features_data['attribute']))
    node_attribute_df = pd.DataFrame({'node_idx': features_data['node_idx'],
                                      'attribute': features_data['attribute']})




    

    return adjacency, node_attribute_dict, node_attribute_df


def computeBalance(communities, node_attributes_dict):
    balanceList = []
    for community in communities:
        red_nodes = [node for node in community if node_attributes_dict[node] == 0]
        blue_nodes = [node for node in community if node_attributes_dict[node] == 1]
        if len(red_nodes) == 0 or len(blue_nodes) == 0:
            balance = 0
        else:
            redRatio = len(red_nodes)/ len(blue_nodes)
            blueRatio = len(blue_nodes)/ len(red_nodes)
            balance = min( redRatio , blueRatio)
        balanceList.append(balance)
    return balanceList

def computeBalanceRedBlue(communities, node_attributes_dict):
    redBalanceList = []
    blueBalanceList = []
    all_nodes = [node for community in communities for node in community]
    all_red_nodes = [node for node in all_nodes if node_attributes_dict[node] == 0]
    all_blue_nodes = [node for node in all_nodes if node_attributes_dict[node] == 1]
    
    phiRed = len(all_red_nodes)/ len(all_nodes)
    phiBlue = len(all_blue_nodes)/ len(all_nodes)
    
    for community in communities:
        red_nodes = [node for node in community if node_attributes_dict[node] == 0]
        blue_nodes = [node for node in community if node_attributes_dict[node] == 1]
        
        redBalance = len(red_nodes)/ len(community)
        blueBalance = len(blue_nodes)/ len(community)
        
        redBalance = redBalance - phiRed
        blueBalance = blueBalance - phiBlue

        redBalanceList.append(redBalance)
        blueBalanceList.append(blueBalance)
    return redBalanceList,blueBalanceList



class NotAPartition(NetworkXError):
    
    """Raised if a given collection is not a partition."""

    def __init__(self, G, collection):
        msg = f"{collection} is not a valid partition of the graph {G}"
        super().__init__(msg)



def modularityCustom(G, communities, weight="weight", resolution=1):

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
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = sum(in_degree[u] for u in comm) if directed else out_degree_sum

        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm
    communityModularityist = []
    for community in communities:
        community_contribution(community)
        communityModularityist.append(community_contribution(community))

    return sum(map(community_contribution, communities)),communityModularityist







def computeMetrics(G, communities,G_attribute):
    modularity = nx.algorithms.community.modularity(G, communities, weight="weight")
    diversitymodularity,diversityModularityList = diversityMetric(G, communities,G_attribute, weight="weight", resolution=1)
    unfairness,unfairnessList,unfairnessModularityPerc,redModularityList,blueModularityList = modularityFairnessMetric(G, communities,G_attribute, weight="weight", resolution=1)
    lUnfairness,lUnfairnessList,lUnfairnessModularityPerc,lRedList,lBlueList = LModularityFairnessMetric(G, communities,G_attribute, weight="weight", resolution=1)
    lDiversity,lDiversityList = LDiversityFairnessMetric(G, communities,G_attribute, weight="weight", resolution=1)
    
    print('\nModularity:',modularity)
    print('---------------------')
    print('RedModularity:',sum(redModularityList),'\tBlueModularity:',sum(blueModularityList))
    print('L-Red Modularity',sum(lRedList),'\tL-Blue Modularity',sum(lBlueList))
    print('---------------------')
    print('Unfairness',unfairness,'\tDiversity:',diversitymodularity)
    print('L-Unfairness:',lUnfairness,'\tL-Diversity:',lDiversity)
    
    return 

def plotCommunities(G, node_attributes_dict, communities, file_name, plotName,method):

    community_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', 'X']

    plt.figure().set_size_inches(22, 19)

    pos = nx.spring_layout(G)


    nx.draw_networkx_edges(G, pos, alpha=0.5)


    for community_id, community_nodes in enumerate(communities):
        community_marker = community_markers[community_id % len(community_markers)]  
        

        node_colors = []
        for node in community_nodes:
            if node_attributes_dict[node] == 0:
                node_colors.append('red')  
            else:
                node_colors.append('blue')  
        

        nx.draw_networkx_nodes(G, pos, nodelist=community_nodes, 
                               node_color=node_colors,  
                               label=f'Community {community_id}',
                               node_size=300, alpha=1, linewidths=1, node_shape=community_marker)  

    nx.draw_networkx_labels(G, pos, font_size=10)


    legend_patches = []
    for community_id in range(len(communities)):
        community_marker = community_markers[community_id % len(community_markers)]
        patch = plt.Line2D([0], [0], marker=community_marker, color='w', label=f'Community {community_id}',
                           markerfacecolor='gray', markersize=10)
        legend_patches.append(patch)


    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Attribute 0 (red)',
                           markerfacecolor='red', markersize=10)
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Attribute 1 (blue)',
                            markerfacecolor='blue', markersize=10)

    plt.legend(handles=legend_patches + [red_patch, blue_patch], loc='best')
    
    plt.savefig('Synth Results\{}\{}\{}.png'.format(method,file_name, plotName))
    plt.close()

    


asymetric_data_path = 'Data//Assymetric'


file_patterns = ['dataset_1000_5_02_09_09_08_08_0' ]

asymetric_file_paths = [os.path.join(asymetric_data_path, file) for file in os.listdir(asymetric_data_path) if 'csv' not in file and any(pattern in file for pattern in file_patterns)]


datasets = []
print('file_paths:',asymetric_file_paths)
for file_path in asymetric_file_paths:
    print('file_path:',file_path)
    
    for file in os.listdir(file_path):
        print(file)
        file = file.split('.')[0]
        if file not in datasets and 'Backup' not in file and 'backup' not in file and 'Original' not in file and 'original' not in file:
            datasets.append(file)



if not os.path.exists('Synth Results'):
    os.makedirs('Synth Results')

    
    
for file_path in datasets:
    datasetName = file_path.split('.')[0]
    datasetName = file_path.split('//')[-1].split('.')[0]
    print('file_path',file_path)
    
    print(datasetName)
    
    
    
    file_path = 'Data//Assymetric//'+file_path.split('.csv')[0]+'\\'
    
    datasetRead = file_path+'\\'+datasetName
    
    adjacency, graph_attributes, node_attribute_df = load_graph_from_files(datasetRead+'.edgelist', datasetRead+'.csv')
    
    graph = nx.Graph(adjacency)
    nx.set_node_attributes(graph, graph_attributes, 'attribute')

    

    
    
 
    
    
    print('\n---GroupDMoN---')
    if not os.path.exists('Synth Results\\Group DMoN Communities'):
        os.makedirs('Synth Results\\Group DMoN Communities')
        
    if not os.path.exists('Synth Results\Group DMoN Communities\\'+datasetName):
        os.makedirs('Synth Results\\Group DMoN Communities\\'+datasetName)
        
        
    blue_communities = groupDMoNClustering(datasetRead+'.edgelist',datasetRead+'.csv',lamda=0.5)
        
    
    

    communities = blue_communities
    print('Number of communities:',len(blue_communities))


    plotName = datasetName+'_Group DMoN'
    plotCommunities(graph, graph_attributes, blue_communities, datasetName, plotName,'Group DMoN Communities')
    
    
    

    
    community_df = pd.DataFrame([(node, community) for community, nodes in enumerate(blue_communities) for node in nodes], columns=['nodes', 'community'])
    

    
    community_df.to_csv(os.path.join('Synth Results\\Group DMoN Communities\\'+datasetName, datasetName + '_communities.csv'), index=False)
    
    
    
    print('\n---Diversity DMoN---')
    if not os.path.exists('Synth Results\\Diversity DMoN Communities'):
        os.makedirs('Synth Results\\Diversity DMoN Communities')
        
    if not os.path.exists('Synth Results\Diversity DMoN Communities\\'+datasetName):
        os.makedirs('Synth Results\\Diversity DMoN Communities\\'+datasetName)
        
        
    blue_communities = diversityDMoNClustering(datasetRead+'.edgelist',datasetRead+'.csv',lamda=0.5)
        
    
    

    communities = blue_communities
    print('Number of communities:',len(blue_communities))
 

    plotName = datasetName+'_Diversity DMoN'
    plotCommunities(graph, graph_attributes, blue_communities, datasetName, plotName,'Diversity DMoN Communities')
    
    
    

    
    community_df = pd.DataFrame([(node, community) for community, nodes in enumerate(blue_communities) for node in nodes], columns=['nodes', 'community'])
    

    
    community_df.to_csv(os.path.join('Synth Results\\Diversity DMoN Communities\\'+datasetName, datasetName + '_communities.csv'), index=False)
    
    
    
    
    computeMetrics(graph, blue_communities, graph_attributes)
        
    
    

    
        

    
    
    
    
    


    
