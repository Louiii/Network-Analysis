import random
import numpy as np
from plot_net import draw_network
from random_graph import make_random_graph
from random_ring_graph import make_ring_graph
#import sys

sample_proportion = 0.2
log = 0

def grp_metric(curr_group, target_groups, m):
    dists = []
    for i in [0,1,2]:
        dists.append( min([( curr_group - target_groups[i] ) % m, ( target_groups[i] - curr_group ) % m]) )
    return min(dists)
    
def groups(group, m):
    return [(group - 1)%m, group, (group + 1)%m]



def find_group(current, target, target_groups, graph, m, k, p, q):
    # look through each neighbour, 
    # if neighbour is in target group: move to it
    # else if it is one away from target group: move to it
    # etc.
    previous_nodes = []
    prob_metric = []
    
    neighbours = graph[current]
    num_neighbours = len(neighbours)
    
    neighbours_queue = list(range(num_neighbours))
    random.shuffle(neighbours_queue)
    
    queries = 0
    for i in neighbours_queue:
        (curr_neighbour_id, curr_group) = neighbours[i]
        
        queries += 1 # each of these is a query
        
        if curr_neighbour_id == target: return (curr_neighbour_id, curr_group), queries
        if curr_group in target_groups: return (curr_neighbour_id, curr_group), queries
        
#        if grp_metric(curr_group, target_groups, m)/m < 0.1:
##            if random.random() < 0.5:
#            return (curr_neighbour_id, curr_group), queries
            
        prob_metric.append( grp_metric(curr_group, target_groups, m) )
        previous_nodes.append( (curr_neighbour_id, curr_group) )
        
    
    indx = prob_metric.index( min(prob_metric) )
    return previous_nodes[ indx ], queries
    
def search_for_target_group(curr, targ, target_groups, graph, m, k, p, q):
    queries = 0
    current = curr[:]
    while current[1] not in target_groups:  
        current, q = find_group(current, targ, target_groups, graph, m, k, p, q)
        queries += q
    return queries, current

def av_search_time_rrgg(graph, m, k, p, q):
    ''' 
    beta is the bias to carry on (if beta is >1)
    random.random()**beta < p  
    
    alpha:
    p is updated s.th if we havent searched many, it has a small p of not carrying on
    this results in us searching more if we havent already
    to change the degree to which this decision is made alpha is used, small alpha for faster decisions
    p *= (num_searched/num_neighbours)**alpha 
    
    p_.. is the value we use for the likihood: metric**pr
    these are used because if we haven't travelled to that node before we want our normal value
    if we have travelled to that node before we want a much smaller value (to make it less likely)
    '''
    alpha, beta = 2, 3# 0.5, 2
    gamma = 1
    average_degree = 3*k*p+(m-3)*k*q
    times = {}
    
    for u in range(m*k):
        for v in range(m*k):
            if u != v and random.random() < sample_proportion:
                curr, targ = (u, u//k), (v, v//k)
                target_groups = groups(targ[1], m)
                #search for target group
#                print('curr: '+str(curr))
                time, current_node = search_for_target_group(curr, targ, target_groups, graph, m, k, p, q)
                if current_node != targ:# must be in the neighbouring group
                    #make random graph from current group and target group
                    rg = {}
                    for tup in graph:
                        #if tup[1] == targ[1] or tup[1] == current_node[1]:
                        if tup[1] in target_groups or tup[1] == current_node[1]:#if node is in current group or target group
                            rg[tup] = graph[tup]
                    rrg = {}
                    for tup in rg:
                        new = []
                        for neigh in rg[tup]:
                            if neigh in rg:
                                new.append(neigh[0])
                        rrg[tup[0]] = new
#                    print(rrg)
                    t = search_rg(current_node[0], v, average_degree, rrg, alpha, beta, gamma, len(rrg))
#                    if t == -1:
#                        t = search_rg(current_node[0], v, average_degree, graph, alpha, beta, gamma, len(graph))
                    time += t
#                time = search_rrgg(curr, targ, average_degree, graph, alpha, beta, m, k, p, q)
                if time in times:
                    times[time] += 1
                else:
                    times[time] = 1
#                print(time)
#            print('DONE')
#        print(avrg(times))
        print(u)
    return times





####################################################################


def carry_on_rg(average_degree, num_neighbours, num_searched, alpha, beta, n):# alpha is the power of how much we've seen
#    print(' av degree: '+str(average_degree)+' num neighbours = ' + str(num_neighbours) + ' num searched = '+str(num_searched) + ' alpha = '+ str(alpha)+' beta = '+ str(beta))
    if num_searched == num_neighbours: 
        return False # searched them all!! choose one!
    if num_searched == 0: 
        return True # searched none! Carry on searching!
    def range_exp_deg(exp_deg, n):#returns my regressor params, for the range above OR below the mean
        return (n*0.02 + 3)+(0.6-n*0.001)*exp_deg-(0.7/n)*exp_deg**2
    def sigmoid(x): # give xeR and sigmoid returns probability e(0,1)
        return 1/(1+np.exp(-x))
    range_ = range_exp_deg(average_degree, n)
    y = (num_neighbours - average_degree)/range_
    p = sigmoid(y) # if num_neighbours is significantly above average degree make p high
#    print('y = '+str(y)+' p = '+str(p))
#    # if num_searched/num_neighbours is high make p higher
#    print('factor' + str((num_searched/num_neighbours)**alpha ))
    p *= ((num_neighbours-num_searched)/num_neighbours)**alpha
    carry_on = random.random()**beta > p
    return carry_on

def choose_next_rg(current, target, graph, average_degree, alpha, beta, gamma, n):
#    print(current)
#    print(graph)
    previous_nodes = []
    prob_metric = []
    neighbours = graph[current]
    num_neighbours = len(neighbours)
    
    if num_neighbours == 1:# if there is only on neighbour...
        return neighbours[0], 0 # move to it

    neighbours_queue = list(range(num_neighbours))
    random.shuffle(neighbours_queue)
    
#    print('num neighbours = '+str(num_neighbours))
    queries = 0
    for i in neighbours_queue:
        curr_neighbour = neighbours[i]
        queries += 1 # each of these is a query
#        print('queries = '+str(queries))
        if curr_neighbour == target:
            return curr_neighbour, queries
        '''
        previous nodes and previous queries are all the nodes we have considered,
        if len(neighbours) is big -> make us more likely to search for longer
        
        after each new vertex is considered; either carry on searching, or pick a node to move to.
        --- carry on with probability proportional to how many neighbours and how many of them left to search.
        -------- if it decides to pick a node: make it slightly more likely to pick a node with higher degree
        '''
        
        if carry_on_rg(average_degree, num_neighbours-1, queries-1, alpha, beta, n) == False:
#            print('ended')
            #PICK NODE!!
            s = sum(prob_metric)
            prob_metric[:] = [x / s for x in prob_metric]
            rand = random.random()
            j = -1
            total = 0
            while total < rand:
                j += 1
                total += prob_metric[j]
            chosen_node = previous_nodes[j]
#            chosen_node = random.choice(previous_nodes)
            return chosen_node, queries

        metric = len(graph[neighbours[i]])
        previous_nodes.append(neighbours[i])
        prob_metric.append(  metric**gamma  )
    
    print(graph)
    print(current)
    print('number of neighbours = '+str(num_neighbours))
    print('DISCONECTED SUBGRAPH ERROR')
    return previous_nodes[0], queries
#        print('nodes:              prob metric:')
#        print(str(previous_nodes)+ ' : '+str(prob_metric))
    
    

def search_rg(current, target, average_degree, graph, alpha, beta, gamma, n):
    queries = 0
#    num_neighbours = len( graph[current] )
#    if num_neighbours == 0:
#        return -1
    while current != target:  
        
        current, q = choose_next_rg(current, target, graph, average_degree, alpha, beta, gamma, n)
#        num_neighbours = len( graph[current] )
#        if num_neighbours > 0:
#            current = c
        
        queries += q
        """Choose next:
        previous_queries.append(  neighbour : degree**1  )
        
        choose from this list with prob proportional based on second value

        if d(neighbour) > average (picking with probability proportional to how high above it is)
        """
    return queries


def plot_(times, g_type, params, n):
    xs, ys = times.keys(), times.values()
    
    av = avrg(times)
    
    import matplotlib.pyplot as plt
    from matplotlib import rcParams, cycler
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis
    rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 3)))
    ax.plot(xs, ys, '.', markersize=5, color=cmap(0.1), label = 'Search time vs frequency for each node u to v')
    ax.plot([av, av], [0, max(times.values())], 'r-', label = 'Average Search Time = '+str(round(av*10)/10))
    ax.legend(fancybox=True, shadow=True)
    xlab = 'Search Time, (number of queries)'
    if log == True:
        xlab = 'Search Time, (log number of queries)'
        plt.xscale('log')
    plt.xlabel(xlab)
    plt.ylabel('Number of Searches')
    plt.title('Search Times for '+g_type+', '+params)
    plt.savefig('search_times_'+params+'.png', dpi=400)

def avrg(times):
    av = 0
    total = 0
    for time in times:
        av += time*times[time]
        total += times[time]
    return float(av) / total









def label_rgg(graph, k):
    new_graph = {}
    for node_id in graph:
        group = node_id//k
        new_graph[(node_id, group)] = [(neighbour, neighbour//k) for neighbour in graph[node_id]]#list(graph[node_id])
    return new_graph


m = 20
k = 20
p = 0.3
q = 0.1
rgg = make_ring_graph(m, k, p, q)

nrg = label_rgg(rgg, k)

times = av_search_time_rrgg(nrg, m, k, p, q)


plot_(times, 'Ring Group Graph', 'm = '+str(m)+', k = '+ str(k)+', p = '+str(p)+', q = '+ str(q), m*k)

    
    
    
    
    
    
    
    
    
    
    
