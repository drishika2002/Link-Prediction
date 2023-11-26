from flask import Flask, render_template, request
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html', num_edges=1)

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

    # Plot the directed graph using Matplotlib
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8, font_color='black', arrowsize=10)
    
    # Save the plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    
    # Encode the image to base64 for displaying in HTML
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')

    return render_template('display_graph.html', graph_image=img_data)

if __name__ == '__main__':
    app.run(debug=True)
