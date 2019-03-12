#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:50:41 2018

@author: louisrobinson
"""
import math

#import platform
from random_ring_graph import make_ring_graph

import plotly
plotly.tools.set_credentials_file(username='Louii', api_key='IJ1ijW3xn1cRI9ynWvkT')
plotly.tools.set_config_file(world_readable=True)
import plotly.plotly as py
import plotly.graph_objs as go

import networkx as nx

def create_edges(graph):
    edges = []
    for u in range(len(graph)):
        all_v = graph[u]
        for v in all_v:
            edges.append( (u, v) )
    return edges

def draw_network(graph):

    G = nx.Graph()#  G is an empty Graph
    
    my_edges = create_edges(graph)
    my_nodes = range(m*k)
    grouped_nodes = []
    for i in range(m): grouped_nodes.append( [i*k + x for x in range(k)] )
    
    
    pos = {}
    # generate m circular points
    for i in range(m):
        TempGraph = nx.Graph()
        TempGraph.add_nodes_from(grouped_nodes[i])
        new_dict = nx.circular_layout(TempGraph)
        #change each elem of new_dict into lists instead of numpy arrays
#        for coords in new_dict:
#            new_dict[coords] = list(new_dict[coords])
        
        t = float(i)/m#goes from 0-1
        r = 4
        displacement = [r*math.sin(t*2*math.pi), r*math.cos(t*2*math.pi)]
        for coords in new_dict:
            for i in [0, 1]:
                new_dict[coords][i] = new_dict[coords][i] + displacement[i]
            new_dict[coords] = list(new_dict[coords])
        pos.update( new_dict )
    
    G.add_nodes_from(my_nodes)
    
    
    G.add_edges_from(my_edges)
        
        
    #########
#    G=nx.random_geometric_graph(200,0.125)
#    pos=nx.get_node_attributes(G,'pos')# {nodeid: [x, y], ...
    
#    dmin=1
#    ncenter=0
#    for n in pos:
#        x,y=pos[n]
#        d=(x-0.5)**2+(y-0.5)**2
#        if d<dmin:
#            ncenter=n
#            dmin=d
    
#    p=nx.single_source_shortest_path_length(G,ncenter)
    
    
    # Create edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
    
    for edge in my_edges:#G.edges():
        x0, y0 = pos[edge[0]][0], pos[edge[0]][1]#G.node[edge[0]]['pos']
        x1, y1 = pos[edge[1]][0], pos[edge[1]][1]#G.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
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
    
    for key in pos:#G.nodes():
        x, y = pos[key][0], pos[key][1]#G.node[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        
        
    # Colour node points
        
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = '# of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])
        
    # Create network graph
        
    fig = go.Figure(data=[edge_trace, node_trace],
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
    
    py.iplot(fig, filename='Ring group graph')
    
    
    
    
    

if __name__ == "__main__":
    m, k = 12, 8
    p, q = 0.495, 0.005
    
    
    graph = make_ring_graph(m, k, p, q)
    # {0:{v1, v2, ..}, 1:{v, ...}, ...}
    
    draw_network(graph)




