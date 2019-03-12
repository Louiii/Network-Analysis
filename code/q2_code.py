import networkx as nx
import networkx.algorithms.approximation as nxaa
import random

def count_edges(graph):
    count = 0
    for i in graph:
        count += len(graph[i])
    return count/2

class PATrial:
    """
    Used when each new node is added in creation of a PA graph.
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are in proportion to the
    probability that it is linked to.
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a PATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities
        
        Returns:
        Set of nodes
        """       
        #compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors
    
def make_complete_graph(num_nodes):
    """Takes the number of nodes num_nodes and returns a dictionary
    corresponding to a complete directed graph with the specified number of
    nodes. A complete graph contains all possible edges subject to the
    restriction that self-loops are not allowed. The nodes of the graph should
    be numbered 0 to num_nodes - 1 when num_nodes is positive. Otherwise, the
    function returns a dictionary corresponding to the empty graph."""
    #initialize empty graph
    complete_graph = {}
    #consider each vertex
    for vertex in range(num_nodes):
        #add vertex with list of neighbours
        complete_graph[vertex] = set([j for j in range(num_nodes) if j != vertex])
    return complete_graph
    
def make_PA_Graph(total_nodes, out_degree):
    """creates a PA_Graph on total_nodes where each vertex is iteratively
    connected to a number of existing nodes equal to out_degree"""
    #initialize graph by creating complete graph and trial object
    PA_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)
    for vertex in range(out_degree, total_nodes):
        PA_graph[vertex] = trial.run_trial(out_degree)
    return PA_graph


def find_edges(graph, vertex):
    """ returns the pairs of nodes not allowed { frozenset([n_1, n_2]), frozenset([n_3, n_2]), ... } """
    neighbours = graph[vertex]
    #now find all of the neighbours neighbours that are also neighbours of vertex
    edges = set([])
    for n in neighbours:
        n_neighbours = graph[n]
        for adj in n_neighbours:
            if adj in neighbours and adj != n:
                edges.add( frozenset([adj, n]) )
    return edges

def b(edges, neighbours):
    # build up a graph
    G = nx.Graph()
    G.add_nodes_from(neighbours)
    G.add_edges_from(edges)
     
    # Independent set
    maximal_iset = nx.maximal_independent_set(G)
    return len(maximal_iset)

def brilliance(graph):
    brilliance_graph = {}
    for vertex in graph:
        edges = find_edges(graph, vertex)
        neighbours  = graph[vertex]
        if len(neighbours) == 0:
            brilliance_graph[vertex] = 0
        else:
            brilliance_graph[vertex] = b(edges, neighbours)
    return brilliance_graph


def load_graph(graph_txt):
    """
    Loads a graph from a text file.
    Then returns the graph as a dictionary.
    
    *Vertices
  1 "Saito, Hiroaki"
  ...
  *Edges
  374 374 5
  ...
    """
    graph = open(graph_txt)
    vertices = False
    edges = False
    node_index = {}
    edges_list = []
    node = 0
    for line in graph:
#        print(line)
        words=[[]]
        if edges == True:
            words = line.split(' ')
#            print(words[2])
#            print(words[3])
#            print(words[4])
#            print(words[4][:-2])
            edges_list.append(  ( int(words[2]), int(words[3]), int(words[4]) )  )
#            print(words)
#            sys.exit()
        words = line.split('"')

        if str(line) == "*Edges\n": 
            vertices = False
            edges = True
#        print(vertices)
        if vertices == True:
#            print('fuc')
            node += 1
            node_index[int(node)] = words[1]
#            
#            sys.exit()
#            
#            node = int(neighbors[0])
#            node_index[node] = set([])
#            for neighbor in neighbors[1 : -1]:
#                node_index[node].add(int(neighbor))
            
        if words[0][0] == "*" and edges == False: 
            vertices = True
    
        
#    print ("Loaded graph with", nodes, "nodes")
    n = len(node_index) + 1#doesn't include 0
    random_graph = {}
    for i in range(1, n + 1):
        print('i: '+str(i)+', of '+str(n+1))
        random_graph[i] = set()
        for edge in edges_list:
            e1, e2 = edge[0], edge[1]
            if i == e1:
                random_graph[i].add( e2 )
            elif i == e2:
                random_graph[i].add( e1 )
    return node_index, random_graph
