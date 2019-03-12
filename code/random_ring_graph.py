import random

def next_to(m, k, node):
    graph = [ [y + x*k for y in range(k)] for x in range(m)]
    group = node // k
    local_groups = [(group-1)%m, group, (group+1)%m]
    local_nodes = []
    for g in local_groups: local_nodes += graph[g]
    return local_nodes

def edge(n2, n1_neighbours):
    if len(n1_neighbours) == 0: return False
    if n2 in n1_neighbours: return True
    return False

def make_ring_graph(m, k, p, q):
    # consider the nodes in a group and the two groups next to that group
    # link each node in this subset with probability p
    # for the complement of this subset link nodes with probability q

    #init
    random_graph = {}
    for i in range(m*k):
        random_graph[i] = set()
        
    for node_u in range(m*k):
        # list all nodes that will be linked with prob p.
        local_nodes = next_to(m, k, node_u)
        
        for node_v in range(node_u, m*k):
            # distict and no edge already
            if node_u != node_v and edge(node_v, random_graph[node_u]) == False: 
                random_number = random.random()
                prob = q
                if node_v in local_nodes:
                    prob = p
                if random_number < prob:
                    random_graph[node_u].add(node_v)
                    random_graph[node_v].add(node_u)
    return random_graph

"""Returns a dictionary to a random graph with the specified number of nodes
and edge probability.  The nodes of the graph are numbered 0 to
num_nodes - 1.  For every pair of nodes, i and j, the pair is considered
twice: once to add an edge (i,j) with probability prob, and then to add an
edge (j,i) with probability prob. 
"""


#
#r = make_random_graph(7, 4, 0.2, 0.05)
#x = [list(r[n]) for n in r]
