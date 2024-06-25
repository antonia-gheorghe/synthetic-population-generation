import dgl
import networkx as nx

# Convert DGL graph to NetworkX graph
nx_graph = dgl.to_networkx(dgl_graph)

# Generate positions for the nodes
pos = nx.spring_layout(nx_graph)  # You can choose different layouts like spring_layout, circular_layout, etc.

# Add positions to the nodes
nx.set_node_attributes(nx_graph, pos, 'pos')

print("here")

# Extract edges and nodes
edge_x = []
edge_y = []

for edge in nx_graph.edges():
    x0, y0 = nx_graph.nodes[edge[0]]['pos']
    x1, y1 = nx_graph.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in nx_graph.nodes():
    x, y = nx_graph.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[str(node) for node in nx_graph.nodes()],
    textposition="bottom center",
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Graph Visualization',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False))
                )

fig.show()