import networkx as nx
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot

import plotly
plotly.tools.set_credentials_file(username='Louii', api_key='IJ1ijW3xn1cRI9ynWvkT')
plotly.tools.set_config_file(world_readable=True)
import plotly.plotly as py
import plotly.graph_objs as go

def create_edges(graph):
    edges = []
    for u in range(len(graph)):
        all_v = graph[u]
        for v in all_v:
            edges.append( (u, v) )
    return edges

def draw_network(graph, n, m):

    G = nx.Graph()#  G is an empty Graph
    
    my_edges = create_edges(graph)
    my_nodes = range(n)
    
    
    G.add_nodes_from(my_nodes)

    pos = nx.random_layout(G)
    #   spring_layout       spectral_layout      fruchterman_reingold_layout
    # bipartite_layout      circular_layout       kamada_kawai_layout 
    #   random_layout       rescale_layout          shell_layout
    
    G.add_nodes_from(my_nodes)
    
    
    G.add_edges_from(my_edges)

    labels=[]
    
    Xn=[pos[k][0] for k in range(len(pos))]
    Yn=[pos[k][1] for k in range(len(pos))]
    
    
    trace_nodes=dict(type='scatter',
                     x=Xn, 
                     y=Yn,
                     mode='markers',
#                     marker=dict(size=6, color='rgb(230,100,256)'),
                     text=labels,
                     hoverinfo='text',
                        marker=dict(
                            showscale=True,
                            # colorscale options
                            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                            colorscale='YlGnBu',
                            reversescale=True,
                            color=[],
                            size=10,
                            colorbar=dict(
                                thickness=15,
                                title='Node Connections',
                                xanchor='left',
                                titleside='right'
                            ),
                            line=dict(width=2)))
    
    Xe=[]
    Ye=[]
    for e in G.edges():
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])

    trace_edges=dict(type='scatter',
                     mode='lines',
                     x=Xe,
                     y=Ye,
                     line=dict(width=0.5, color='rgb(25,25,25)'),
                     hoverinfo='none' 
                    )
        
        
    # Colour node points
        
    for node, adjacencies in enumerate(G.adjacency()):
        trace_nodes['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = '# of connections: '+str(len(adjacencies[1]))+', node: '+str(node)
        trace_nodes['text']+=tuple([node_info])
        
    # Create network graph
        
    fig = go.Figure(data=[trace_edges, trace_nodes],
                 layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    plot(fig)









