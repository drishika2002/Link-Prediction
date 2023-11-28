from flask import Flask, render_template, request
import networkx as nx
import matplotlib.pyplot as plt
import io
import numpy as np
import base64
import pickle
import pandas as pd
from pandas import HDFStore

# Switch Matplotlib to a non-interactive backend
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.static_folder = 'static'

# Load the XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    xgboost_model = pickle.load(file)

# Load features from H5 file
hdf_train = HDFStore('storage_sample_stage2.h5')
df_final_train = hdf_train.get('train_df')
hdf_train.close()

hdf_test = HDFStore('storage_sample_stage2.h5')
df_final_test = hdf_test.get('test_df')
hdf_test.close()

@app.route('/')
def index():
    return render_template('index.html', num_edges=1)

def jaccard_distance_followees(graph, node_x, node_y):
    set_followees_x = set(graph.predecessors(node_x))
    set_followees_y = set(graph.predecessors(node_y))

    if len(set_followees_x.union(set_followees_y)) == 0:
        return 0.0
    else:
        jaccard_distance = 1 - len(set_followees_x.intersection(set_followees_y)) / len(set_followees_x.union(set_followees_y))
        return jaccard_distance

def jaccard_distance_followers(graph, node_x, node_y):
    set_followers_x = set(graph.successors(node_x))
    set_followers_y = set(graph.successors(node_y))

    if len(set_followers_x.union(set_followers_y)) == 0:
        return 0.0
    else:
        jaccard_distance = 1 - len(set_followers_x.intersection(set_followers_y)) / len(set_followers_x.union(set_followers_y))
        return jaccard_distance

def adar_index(g, a, b):
    sum = 0
    try:
        n = list(set(g.successors(a)).intersection(set(g.successors(b))))
        if len(n) != 0:
            for i in n:
                sum = sum + (1 / np.log10(len(list(g.predecessors(i)))))
            return sum
        else:
            return 0
    except:
        return 0
    
def compute_shortest_path_length(g, a, b):
    p = -1
    try:
        if g.has_edge(a, b):
            g.remove_edge(a, b)
            p = nx.shortest_path_length(g, source=a, target=b)
            g.add_edge(a, b)
        else:
            p = nx.shortest_path_length(g, source=a, target=b)
        return p
    except:
        return -1
    
def follows_back(g, a, b):
    if g.has_edge(b, a):
        return 1
    else:
        return 0
    
def calculate_jaccard_distance(g, source_node, destination_node, distance_function):
    if source_node not in g or destination_node not in g:
        return 0  # or handle missing nodes in a way that makes sense for your use case
    else:
        return distance_function(g, source_node, destination_node)

def extract_features(G, node1, node2):
    jaccard_followers = calculate_jaccard_distance(G, node1, node2, jaccard_distance_followers)
    jaccard_followees = calculate_jaccard_distance(G, node1, node2, jaccard_distance_followees)
    shortest_path_length = compute_shortest_path_length(G, node1, node2)
    adar_index_value = adar_index(G, node1, node2)
    follows_back_value = follows_back(G, node1, node2)
    features = np.array([jaccard_followers, jaccard_followees, shortest_path_length, adar_index_value, follows_back_value])
    return features

@app.route('/display_graph', methods=['POST'])
def display_graph():
    num_edges = int(request.form['num_edges'])
    source_nodes = []
    destination_nodes = []

    for i in range(num_edges):
        source_key = f'source{i}'
        destination_key = f'destination{i}'

        if source_key in request.form and destination_key in request.form:
            source = request.form[source_key]
            destination = request.form[destination_key]
            source_nodes.append(source)
            destination_nodes.append(destination)

    # Create directed graph using NetworkX
    G = nx.DiGraph()
    for source, destination in zip(source_nodes, destination_nodes):
        G.add_edge(source, destination)

    # Predict for all non-existing edges
    predictions = {}
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2 and not G.has_edge(node1, node2):
                features = extract_features(G, node1, node2)

                # Accessing values from the DataFrame with error handling
                source_rows = df_final_train[df_final_train['source_node'] == node1]
                destination_rows = df_final_train[df_final_train['destination_node'] == node2]

                if not source_rows.empty and not destination_rows.empty:
                    features = extract_features(G, node1, node2)
                    edge_score = xgboost_model.predict_proba(features.reshape(1, -1))[:, 1].item()
                    threshold = 0.5
                    prediction = edge_score > threshold

                    predictions[(node1, node2)] = prediction

    # Plot the directed graph using Matplotlib
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8, font_color='black', arrowsize=10)

    # Display the predictions for non-existing edges
    for (node1, node2), prediction in predictions.items():
        plt.annotate(f'Edge: {node1} -> {node2}, Prediction: {"Will" if prediction else "Will Not"} be formed',
                     xy=pos[node1], xytext=pos[node2], arrowprops=dict(arrowstyle="->", color='red'))

    # Save the plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Encode the image to base64 for displaying in HTML
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')

    return render_template('display_graph.html', graph_image=img_data)

if __name__ == '__main__':
    app.run(debug=True)
