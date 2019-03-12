import math

#import platform
from random_ring_graph import make_ring_graph
from brilliance import br



#import plotly
import networkx as nx
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import plotly.graph_objs as go
#init_notebook_mode(connected=True)
#import plotly.graph_objs as go

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
        t = float(i)/m#goes from 0-1
        r = 2.5
        displacement = [r*math.sin(t*2*math.pi), r*math.cos(t*2*math.pi)]
        for coords in new_dict:
            new_dict[coords] = 0.5*new_dict[coords] + displacement
        pos.update( new_dict )
    
    G.add_nodes_from(my_nodes)
    
    
    G.add_edges_from(my_edges)
    # make m circular graphs for the k nodes in each group
    # transpose and concat all their positions
    
    #pos = nx.fruchterman_reingold_layout(G)   
    #   spring_layout       spectral_layout      fruchterman_reingold_layout
    # bipartite_layout      circular_layout       kamada_kawai_layout 
    #   random_layout       rescale_layout          shell_layout
    
    #{0: array([ 0.07948263, -0.0853657 ]), 1: array([ 0.14558215, -0.02568524]),..
    
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
#    trace_nodes=dict(type='scatter',
#                     x=Xn, 
#                     y=Yn,
#                     mode='markers',
#                     marker=dict(size=6, color='rgb(230,100,256)'),
#                     text=labels,
#                     hoverinfo='text')
    
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
#    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
#              zeroline=False,
#              showgrid=False,
#              showticklabels=False,
#              title='' 
#              )
#    
#    
#    
#    title_ = 'Network: m= '+str(m)+' groups, k = '+str(k)+' nodes in group'
#    layout=dict(title= title_,  
#                font= dict(family='Balto'),
#                width=600,
#                height=600,
#                autosize=False,
#                showlegend=False,
#                xaxis=axis,
#                yaxis=axis,
#                margin=dict(
#                l=40,
#                r=40,
#                b=85,
#                t=100,
#                pad=0,
#        ),
#        hovermode='closest',
#        plot_bgcolor='#efecea', #set background color            
#        )
#    fig = dict(data=[trace_edges, trace_nodes], layout=layout)
#    
#    
    for node, adjacencies in enumerate(G.adjacency()):
        trace_nodes['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = '# of connections: '+str(len(adjacencies[1]))+', node: '+str(node)
        trace_nodes['text']+=tuple([node_info])
        
    # Create network graph
    t = '<br>Ring Group Graph: Groups = ' + str(m)+', Nodes per group = ' + str(k)
    
    fig = go.Figure(data=[trace_edges, trace_nodes],
                 layout=go.Layout(
                    title=t,
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



if __name__ == "__main__":
    m, k = 12, 6
    p, q = 0.495, 0.005
    
    # p + q = 0.5
    # investigate diameter
    graph = make_ring_graph(m, k, p, q)
    brilliance = br(graph)
    # {0:{v1, v2, ..}, 1:{v, ...}, ...}
    
#    graph = {0:[2, 5, 9, 11, 13],1:[7, 9, 12, 13],2:[0,6,7,11],3:[4,8,10,12],4:[3,9],5:[0,7,12],6:[2],7:[1,2,5],8:[3],9:[0,1,4],10:[3],11:[0,2],12:[1,3,5],13:[0,1]}
    
    draw_network(graph)







