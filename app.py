from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static' 

@app.route('/')
def index():
    return render_template('index.html', num_edges=1)

@app.route('/display_graph', methods=['POST'])
def display_graph():
    num_edges = int(request.form['num_edges'])
    edges = []

    for i in range(num_edges):
        x = float(request.form[f'x{i}'])
        y = float(request.form[f'y{i}'])
        edges.append((x, y))

    return render_template('display_graph.html', num_edges=num_edges, edges=edges)

if __name__ == '__main__':
    app.run(debug=True)

