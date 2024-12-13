import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from run_cascade import functions_data_transformation as transform
from batch_process import gui_configurations as configurations



from run_cascade import functions_data_transformation as transform
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import multiprocessing

def load_for_networkx(data_folder):  ## creates a dictionary for the suite2p paths in the given data folder (e.g.: folder for well_x)
    """
    Creates a dictionary for networkx analysis from the SUITE2P_STRUCTURE in the data folder.
    """    
    stat = transform.load_npy_array(os.path.join(data_folder, *transform.SUITE2P_STRUCTURE["stat"]))
    cascade_predictions = transform.load_npy_array(os.path.join(data_folder, *transform.SUITE2P_STRUCTURE["cascade_predictions"]))
    iscell = transform.load_npy_array(os.path.join(data_folder, *transform.SUITE2P_STRUCTURE['iscell']))[:,0].astype(bool)
    neuron_data = {}

    for idx, neuron_stat in enumerate(stat):
        x_median = np.median(neuron_stat['xpix'])
        y_median = np.median(neuron_stat['ypix'])
        predicted_spikes = cascade_predictions[idx,:]

        neuron_data[f"neuron_{idx}"] = {
            "x": x_median,
            "y": y_median,
            "predicted_spikes": predicted_spikes,
            "IsUsed": iscell[idx]
        }
    filtered_neuron_data = {}

    for key, value in neuron_data.items():
        if value["IsUsed"]:
            filtered_neuron_data[key] = value
    
    return  filtered_neuron_data

def create_template_matrix(neuron_data):
    num_neurons = len(neuron_data)
    temp_matrix = np.random.rand(num_neurons, num_neurons)
    temp_matrix[temp_matrix<0.9] = 0
    np.fill_diagonal(temp_matrix, 0)
    G = nx.from_numpy_array(temp_matrix)
    mapping = {i: neuron_id for i, neuron_id in enumerate(neuron_data.keys())} #rename index from neuron_0... for networkx
    G = nx.relabel_nodes(G, mapping)
    return G

def getImg(ops):
    """Accesses suite2p ops file (itemized) and pulls out a composite image to map ROIs onto"""
    Img = ops["meanImg"] # Also "max_proj", "meanImg", "meanImgE"
    mimg = Img # Use suite-2p source-code naming
    mimg1 = np.percentile(mimg,1)
    mimg99 = np.percentile(mimg,99)
    mimg = (mimg - mimg1) / (mimg99 - mimg1)
    mimg = np.clip(mimg, 0, 1)
    mimg *= 255
    mimg = mimg.astype(np.uint8)
    return mimg

def extract_and_plot_neuron_connections(node_graph, neuron_data, data_folder, sample_name, ops):
    # Prepare image
    mimg = getImg(ops)
    plt.figure(figsize=(20, 20))
    
    # Display the mean image
    plt.imshow(mimg, cmap='gray', interpolation='nearest')
    plt.title(f"Sample: {sample_name} - Overlayed Communities", fontsize=24)
    
    # Prepare graph layout
    for neuron_id, data in neuron_data.items():
        node_graph.add_node(neuron_id, pos=(data['x'], data['y']))
    pos = nx.get_node_attributes(node_graph, 'pos')
    
    # Community Detection
    neuron_clubs = list(greedy_modularity_communities(node_graph))
    community_map = {
        node: community_idx
        for community_idx, community in enumerate(neuron_clubs)
        for node in community
    }
    communities = list(range(len(neuron_clubs)))
    community_spikes = {
        community: np.zeros_like(next(iter(neuron_data.values()))['predicted_spikes'])
        for community in communities
    }
    for node, community_idx in community_map.items():
        community_spikes[community_idx] += np.nan_to_num(neuron_data[node]['predicted_spikes'])
    
  
    # Node statistics
    node_degree_dict = dict(node_graph.degree)
    clustering_coeff_dict = nx.clustering(node_graph)
    betweenness_centrality_dict = nx.betweenness_centrality(node_graph)
    try:
        eigenvector_centrality_dict = nx.eigenvector_centrality(node_graph)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality_dict = {node: None for node in node_graph.nodes}
    
    # Edge Statistics
    edge_data = []
    for (u, v, data) in node_graph.edges(data=True):
        edge_data.append({
            'source': u,
            "target": v,
            'weight': data.get("weight", 1),
        })
    
    community_sizes = {community_idx: len(community) for community_idx, community in enumerate(neuron_clubs)}
    raw_data = []
    for node, neuron in zip(node_graph.nodes, neuron_data):
        raw_data.append({
            "neuron_id": node,
            "x": neuron_data[node]["x"],
            "y": neuron_data[node]["y"],
            "community": community_map[node],
            "community_size": community_sizes[community_map[node]],
            "degree": node_degree_dict[node],
            "clustering_coefficient": clustering_coeff_dict[node],
            "betweenness_centrality": betweenness_centrality_dict[node],
            "eigenvector_centrality": eigenvector_centrality_dict[node],
            "total_predicted_spikes": np.nansum(neuron_data[neuron]['predicted_spikes']),
            "avg_predicted_spikes": np.nanmean(neuron_data[neuron]['predicted_spikes'])
        })
    df_nodes = pd.DataFrame(raw_data)
    # df_nodes["community_spikes"]
    df_nodes.to_csv(os.path.join(data_folder, f"{sample_name}_graph_node_data.csv"), index=False)
    
    df_edges = pd.DataFrame(edge_data)
    df_edges.to_csv(os.path.join(data_folder, f"{sample_name}_graph_edge_data.csv"), index=False)
    
    # Overlay graph on the image
    community_colors = [community_map[node] for node in node_graph.nodes]
    cmap = matplotlib.cm.get_cmap('tab10')
    community_ids = sorted(set(community_map.values()))
    colors = cmap.colors[:len(community_ids)]
    community_color_map = {community_id: colors[i] for i, community_id in enumerate(community_ids)}
    community_colors = [community_map[node] for node in node_graph.nodes]
    
    nx.draw_networkx_nodes(
        node_graph,
        pos=pos,
        node_size=100,
        node_color=community_colors,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, f"{sample_name}_networkx_image_overlay.png"))
    plt.close()
    plt.figure(figsize=(10,6))
    communities = list(community_spikes.keys())
    total_spikes = list(community_spikes.values())
    bar_colors = [community_color_map[community] for community in communities]
    plt.bar(communities, total_spikes, color=bar_colors)
    plt.xlabel('Community', fontsize=14)
    plt.ylabel('Total Predicted Spikes', fontsize=14)
    plt.title(f"Total Predicted Spikes per Community - {sample_name}", fontsize=16)
    plt.xticks(communities)
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, f"{sample_name}_total_spikes_per_community.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for community, spikes in community_spikes.items():
        plt.plot(spikes, label = f'{community}', color = community_color_map[community])
    plt.xlabel('Frame', fontsize=14)
    plt.ylabel('Total Predicted Spikes', fontsize=14)
    plt.title(f"Total Predicted Spikes per Community (Line Plot) - {sample_name}", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, f"{sample_name}_total_spikes_line_plot.png"))
    plt.close()

def test_extract_and_plot_neuron_connections(node_graph, neuron_data, data_folder, sample_name, ops):
    # Prepare image
    mimg = getImg(ops)
    # plt.figure(figsize=(20, 20))
    
    # Display the mean image
    # plt.imshow(mimg, cmap='gray', interpolation='nearest')
    # plt.title(f"Sample: {sample_name} - Overlayed Communities", fontsize=24)
    
    # Prepare graph layout
    for neuron_id, data in neuron_data.items():
        node_graph.add_node(neuron_id, pos=(data['x'], data['y']))
    pos = nx.get_node_attributes(node_graph, 'pos')
    
    # Community Detection
    neuron_clubs = list(greedy_modularity_communities(node_graph))
    community_map = {
        node: community_idx
        for community_idx, community in enumerate(neuron_clubs)
        for node in community
    }
    communities = list(range(len(neuron_clubs)))
    community_spikes = {
        community: np.zeros_like(next(iter(neuron_data.values()))['predicted_spikes'])
        for community in communities
    }
    for node, community_idx in community_map.items():
        community_spikes[community_idx] += np.nan_to_num(neuron_data[node]['predicted_spikes'])
    
   # Node statistics
    node_degree_dict = dict(node_graph.degree)
    clustering_coeff_dict = nx.clustering(node_graph)
    betweenness_centrality_dict = nx.betweenness_centrality(node_graph)
    try:
        eigenvector_centrality_dict = nx.eigenvector_centrality(node_graph)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality_dict = {node: None for node in node_graph.nodes}
    
    # Edge Statistics
    edge_data = []
    for (u, v, data) in node_graph.edges(data=True):
        edge_data.append({
            'source': u,
            "target": v,
            'weight': data.get("weight", 1),
        })
    
    community_sizes = {community_idx: len(community) for community_idx, community in enumerate(neuron_clubs)}
    raw_data = []
    for node, neuron in zip(node_graph.nodes, neuron_data):
        raw_data.append({
            "neuron_id": node,
            "x": neuron_data[node]["x"],
            "y": neuron_data[node]["y"],
            "community": community_map[node],
            "community_size": community_sizes[community_map[node]],
            "degree": node_degree_dict[node],
            "clustering_coefficient": clustering_coeff_dict[node],
            "betweenness_centrality": betweenness_centrality_dict[node],
            "eigenvector_centrality": eigenvector_centrality_dict[node],
            "total_predicted_spikes": np.nansum(neuron_data[neuron]['predicted_spikes']),
            "avg_predicted_spikes": np.nanmean(neuron_data[neuron]['predicted_spikes'])
        })
    df_nodes = pd.DataFrame(raw_data)
    # df_nodes["community_spikes"]
    # df_nodes.to_csv(os.path.join(data_folder, f"{sample_name}_graph_node_data.csv"), index=False)
    
    df_edges = pd.DataFrame(edge_data)
    # df_edges.to_csv(os.path.join(data_folder, f"{sample_name}_graph_edge_data.csv"), index=False)
    

    communities = list(community_spikes.keys())
 
    return community_spikes, communities, df_edges, df_nodes


    
def plot_neuron_connections(data_folder):
    print('extracting neuron data for network x')
    neuron_data = load_for_networkx(data_folder)
    print("creating networkx node graph")
    node_graph = create_template_matrix(neuron_data)
    sample_name = os.path.basename(data_folder)
    print(sample_name)
    ops = transform.load_npy_array(os.path.join(data_folder, *transform.SUITE2P_STRUCTURE["ops"])).item()
    community_stats, communities = test_extract_and_plot_neuron_connections(node_graph, neuron_data, data_folder, sample_name, ops)
    return community_stats, communities

def main():
    for sample in transform.get_file_name_list(configurations.main_folder, file_ending = 'samples', supress_printing=False):
        print(f"Processing {sample}")
        plot_neuron_connections(sample)
        print('Finished processing')

if __name__ == '__main__':
    main()