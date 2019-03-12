import random

def make_random_graph(num_nodes, prob):
    """Returns a dictionary to a random graph with the specified number of nodes
    and edge probability.  The nodes of the graph are numbered 0 to
    num_nodes - 1.  For every pair of nodes, i and j, the pair is considered
    twice: once to add an edge (i,j) with probability prob, and then to add an
    edge (j,i) with probability prob. 
    """
    #initialize empty graph
    random_graph = {}
    #consider each vertex
    for vertex in range(num_nodes):
        random_graph[vertex] = set()
    for vertex in range(num_nodes):
        for neighbour in range(vertex, num_nodes):
            if vertex != neighbour:
                random_number = random.random()
                if random_number < prob:
                    random_graph[vertex].add(neighbour)
                    random_graph[neighbour].add(vertex)
    return random_graph
