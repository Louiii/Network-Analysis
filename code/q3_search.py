import random
import numpy as np
from plot_net import draw_network
from random_graph import make_random_graph
from random_ring_graph import make_ring_graph
#import sys

sample_proportion = 0.2
log = 0



def carry_on_rg(average_degree, num_neighbours, num_searched, alpha, beta, n):# alpha is the power of how much we've seen
#    print(' av degree: '+str(average_degree)+' num neighbours = ' + str(num_neighbours) + ' num searched = '+str(num_searched) + ' alpha = '+ str(alpha)+' beta = '+ str(beta))
    if num_searched == num_neighbours: 
        return False # searched them all!! choose one!
    if num_searched == 0: 
        return True # searched none! Carry on searching!
    def range_exp_deg(exp_deg, n):#uses my regressor params to estimate the range either side of the average
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
            print('DONE')
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
#        print('nodes:              prob metric:')
#        print(str(previous_nodes)+ ' : '+str(prob_metric))
    

def search_rg(current, target, average_degree, graph, alpha, beta, gamma, n):
    queries = 0
    i=0
    while current != target:  
        current, q = choose_next_rg(current, target, graph, average_degree, alpha, beta, gamma, n)
        queries += q
        i+=1
        """Choose next:
        previous_queries.append(  neighbour : degree**1  )
        
        choose from this list with prob proportional based on second value

        if d(neighbour) > average (picking with probability proportional to how high above it is)
        """
    print('average_queries = '+str(queries/i))
    return queries

def av_search_time_rg(graph, n, p, alpha, beta, gamma):
    ''' 
    beta is the bias to carry on (if beta is >1)
    random.random()**beta < p  
    
    alpha:
    p is updated s.th if we havent searched many, it has a small p of not carrying on
    this results in us searching more if we havent already
    to change the degree to which this decision is made alpha is used, small alpha for faster decisions
    p *= (num_searched/num_neighbours)**alpha 
    
    '''
    average_degree = n*p
    times = {}
    for u in range(n):
        for v in range(n):
            if u != v and random.random() < sample_proportion:
                time = search_rg(u, v, average_degree, graph, alpha, beta, gamma, n)
                if time in times:
                    times[time] += 1
                else:
                    times[time] = 1
#                print('\nnext search\n')
#        print(u)
    return times



def plot_(times, g_type, params, n):
    xs, ys = times.keys(), times.values()
    
    av = avrg(times, n)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    ax.plot(xs, ys, '.', markersize=5, label = 'Search time vs frequency for each node u to v')
    ax.plot([av, av], [0, max(times.values())], 'r-', label = 'Average Search Time = '+str(round(av*10)/10))
    ax.legend(fancybox=True, shadow=True)
    xlab = 'Search Time, (number of queries)'
    if log == True:
        xlab = 'Search Time, (log number of queries)'
        plt.xscale('log')
    plt.xlabel(xlab)
    plt.ylabel('Number of Searches')
    plt.title('Search Times for '+g_type+', '+params)
    plt.savefig('search_times_'+params+'.png', dpi=400, bbox_inches = 'tight')

def avrg(times, n):
    av = 0
    total = 0
    for time in times:
        av += time*times[time]
        total += times[time]
    return float(av) / total
    

#    g = {0:[2, 5, 9, 11, 13, 15],1:[7, 9, 12, 13, 15],2:[0,6,7,11, 15],3:[4,8,10,12, 14],4:[3,9, 14],5:[0,7,12, 15],
#         6:[2],7:[1,2,5],8:[3, 14],9:[0,1,4, 15],10:[3, 14],11:[0,2, 14, 15],12:[1,3,5],13:[0,1, 15], 14:[3,4,8,10,11], 15:[0,1,2,5,9,11,13]}
'''small alpha for faster decisions

beta is the net bias to carry on large beta for longer decisions
random.random()**beta < p  
'''
alpha = 2
beta = 0.8# 0.5, 2
gamma = 1

    
n = 100
p = 0.2
g = make_random_graph(n, p)

for i in g:
    g[i]=list(g[i])


times = av_search_time_rg(g, len(g), p, alpha, beta, gamma)

plot_( times, 'Random Graph', 'n = '+str(len(g))+', p = '+ str(p) , len(g))



#    betas = [0.5, 1, 1.5]
#    alphas = [0.5, 1, 1.5]
#    gammas = [0.5, 1, 1.5]


#    a=0
#    for i in g:
#        a += len(g[i])
#    p = a/(len(g)*(len(g)-1))

#    avs = {}
#    for a in alphas:
#        for b in betas:
#            for ga in gammas:
#                times = av_search_time_rg(g, len(g), p, a, b, ga)
#                avs[(a, b, ga)] = avrg(times, len(g)) 
#    for i in avs:
#        print('Average search time : ' + str(avs[i])+' , betas = '+str(i))

#    draw_network(g, len(g), 0)