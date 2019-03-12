from random_graph import make_random_graph
from random_ring_graph import make_ring_graph
import queue
#import math


def compute_in_degrees(graph):
    """Takes a directed graph and computes the in-degrees for the nodes in the
    graph. Returns a dictionary with the same set of keys (nodes) and the
    values are the in-degrees."""
    #initialize in-degrees dictionary with zero values for all vertices
    in_degree = {}
    for vertex in graph:
        in_degree[vertex] = 0
    #consider each vertex
    for vertex in graph:
        #amend in_degree[w] for each outgoing edge from v to w
        for neighbour in graph[vertex]:
            in_degree[neighbour] += 1
    return in_degree

def in_degree_distribution(graph):
    """Takes a directed graph and computes the unnormalized distribution of the
    in-degrees of the graph.  Returns a dictionary whose keys correspond to
    in-degrees of nodes in the graph and values are the number of nodes with
    that in-degree. In-degrees with no corresponding nodes in the graph are not
    included in the dictionary."""
    #find in_degrees
    in_degree = compute_in_degrees(graph)
    #initialize dictionary for degree distribution
    degree_distribution = {}
    #consider each vertex
    for vertex in in_degree:
        #update degree_distribution
        if in_degree[vertex] in degree_distribution:
            degree_distribution[in_degree[vertex]] += 1
        else:
            degree_distribution[in_degree[vertex]] = 1
    return degree_distribution

def max_dist(graph, source):
    """finds the distance (the length of the shortest path) from the source to
    every other vertex in the same component using breadth-first search, and
    returns the value of the largest distance found"""
    q = queue.Queue()
    found = {}
    distance = {}
    for vertex in graph: 
        found[vertex] = 0
        distance[vertex] = -1
    max_distance = 0
    found[source] = 1
    distance[source] = 0
    q.put(source)
    while q.empty() == False:
        current = q.get()
        for neighbour in graph[current]:
            if found[neighbour] == 0:
                found[neighbour] = 1
                distance[neighbour] = distance[current] + 1
                max_distance = distance[neighbour]
                q.put(neighbour)
    return max_distance
    
def diameter(graph):
    """returns the diameter of a graph by finding greatest max distance"""
    return max([max_dist(graph, source) for source in range(len(graph)) ])

        
def diameter_vs_prob(k, q, probs):
    """diameter and clustering coefficient vs rewiring prob with k trials"""
    xdata = []
    ydata = []
#    zdata = []
#    prob = 0.0005
#    while prob < 0.5:
    for prob in probs:
        print (prob)
        xdata += [prob]
        diameters = []
#        coeffs = []
        for i in range(k):
            graph = make_ring_graph(20, 20, prob, q)
            diameters += [diameter(graph)]
#            coeffs += [clustering_coefficient(graph)]
        ydata += [sum(diameters) / k ] #divide by 19 as this diameter of circle lattice
#        zdata += [sum(coeffs) /  k / 0.7] #divide by 0.7 as this is clustering coefficient of circle lattice
#        prob = 1.2*prob#*1.1
    return xdata, ydata#, zdata

def find_range(x, y):#{degree:proportion, ..}
    '''find the r1 where P(X<d) = 0.05 and r2 where P(X<d) = 0.95'''
    degree_distribution = {}
    for i in range(len(x)):
        degree_distribution[x[i]] = y[i]
      
    p = 0
    for d in degree_distribution:
        p+=d*degree_distribution[d]
    print( 'p = ' + str(p) )
    y = [y_i/p for y_i in y]
    for i in range(len(x)):
        degree_distribution[x[i]] = y[i]
        
    p = 0
    r = []
    check = False
    previous_d = list(degree_distribution.keys())[0]
    for d in degree_distribution:
        p += d*degree_distribution[d]
#        print('d='+str(d)+'previous_d='+str(previous_d))
        print('p='+str(p))
        if p > 0.05 and check == False:
            if previous_d == d:
                previous_d = d - 1
                prev_y = 0
            else:
                prev_y = degree_distribution[previous_d]
            dx = d-previous_d
            dy = degree_distribution[d] - prev_y
            dp = p - 0.05
            check = True
            r.append( previous_d + dp*dy/dx )
        if p > 0.95:
            print('check')
            dx = d-previous_d
            dy = degree_distribution[d] - degree_distribution[previous_d]
            dp = p - 0.95
            check = True
            r.append( previous_d + dp*dy/dx )
            return r
        previous_d = d
    
###################################################################
'''RANDOM RING GRAPHS'''
m, k = 20, 20
rrg = [(m,k,0.5 - x*0.05) for x in range(11)]#(m, k, p, q)
xs, ys = [], []
exps = []
ranges = []
for b in rrg:
    d_size = 600
    d = {}
    for i in range(d_size): d[i] = 0
    a=20
    for i in range(a):
        g = make_ring_graph(b[0], b[1], b[2], 0.5-b[2])
        dg = in_degree_distribution(g)
        for i in range(d_size): 
            if i in dg.keys():
                d[i] += float(dg[i])/a
    total = 0
    for i in range(d_size): 
        if d[i] == 0:
            del d[i]
        else:
            total += float(i*d[i])/(b[0]*b[1])
    exps.append(total)
    x, y = d.keys(), d.values()
    x = [ float(x_i)*b[2]**0 for x_i in x]
    y = [ float(y_i) / (b[0]*b[1]) for y_i in y]
    ranges.append( find_range(x, y) )
#    x = list(x)
#    y = list(y)
    xs.append(x)
    ys.append(y)

table = []
for i in range(len(exps)):
    row = [exps[i], ranges[i][0], ranges[i][1], ranges[i][1]-ranges[i][0]]
    table.append(row) 
#import csv
#
## write it
#with open('test_file.csv', 'w') as csvfile:
#    writer = csv.writer(csvfile)
#    [writer.writerow(r) for r in table]
#    
## read it
#with open('test_file.csv', 'r') as csvfile:
#    reader = csv.reader(csvfile)
#    table = [[float(e) for e in r] for r in reader]

table = [(table[i][0], table[i][3]) for i in range(6)]
x, y = zip(*table)

##### RANGES CODE
import matplotlib.pyplot as plt
x_new = [i*0.5 for i in range(41, 220)]
import numpy.polynomial.polynomial as poly

coefs = poly.polyfit(x, y, 2)
ffit = poly.polyval(x_new, coefs)

fig, ax = plt.subplots()
ax.plot(x_new, ffit, 'k-', label='poly. reg.='+str(round((coefs[0])*100)/100)+'+'+str(round((coefs[1])*1000)/1000)+'$x$+'+str(round((coefs[2])*100000)/100000)+'$x^2$')
ax.plot(x, y, 'rx', markersize=10, label='ranges and $\overline{d}$ for ring graph with m=20, k=20')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True)
plt.title('Degree Distribution')
plt.xlabel('Expected degree of node, $\overline{d}$')
plt.ylabel('Range of distribution')
plt.plot()
plt.savefig('exps_ranges.png',dpi=300, bbox_inches = 'tight')
######

import numpy as np
from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
l=[]
for i in range(len(rrg)):
    l.append('p = '+str(round((rrg[i][2])*100)/100)+', q = '+str(round((0.5-rrg[i][2])*100)/100)+', $\overline{d}$ = '+str(round(exps[i]*10)/10) )#l.append('m = '+str(plott[0])+', k = '+str(plott[1])+', p = '+str(plott[2])+', q = '+str(0.5-plott[2]))
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(rrg))))
for i in range(len(rrg)):
    ax.plot(xs[i], ys[i], color=cmap((i)*(1/len(rrg))), marker='.', label=l[i])
    ax.plot([exps[i], exps[i]], [0, 0.12], 'k--')
ax.legend(loc='right center', fancybox=True, shadow=True)
plt.xlabel('Degree')
plt.ylabel('Normalised Rate')
plt.title('Degree Distribution, m = '+str(m)+', k = '+str(k))
plt.plot()
plt.savefig('Degree_Distribution.png',dpi=500, bbox_inches = 'tight')

###################################################################
'''RANDOM GRAPHS'''

import matplotlib.pyplot as plt
rg = [(100, 0.01), (1000, 0.001), (10000, 0.0001)]
xs, ys = [], []
for b in rg:
    g = make_random_graph(b[0], b[1])
    d = in_degree_distribution(g)
    x, y = d.keys(), d.values()
    y = [ float(y_i) / b[0] for y_i in y]
    xs.append(list(x))
    ys.append(y)

#plt.clf()
prod = 0
for i in range(3):
    for j in range(len(xs[i])):
        prod += xs[i][j]*ys[i][j]
prod /= 3


fig, ax = plt.subplots()
ax.plot(xs[0], ys[0], 'b-', label='N = 100, p = 0.01')
ax.plot(xs[1], ys[1], 'c-', label='N = 1000, p = 0.001')
ax.plot(xs[2], ys[2], 'm-', label='N = 10000, p = 0.0001')
ax.plot([prod, prod], [0, 1], 'r-', label='Expected degree')
ax.legend()
#plt.style.use('classic')
plt.xlabel('Degree')
plt.ylabel('Normalised Rate')
plt.title('Degree Distribution')
#leg = ax.legend();
plt.plot()

plt.savefig('random_Degree_Distribution.png')

###################################################################
'''RING GRAPHS CHECK'''

import matplotlib.pyplot as plt
#rg = [(100, 0.01), (1000, 0.001), (10000, 0.0001)]
rg = [(20, 20, 0.0025), (30, 30, 0.001111), (40, 40, 0.000625), ]
xs, ys = [], []
for b in rg:
#    g = make_random_graph(b[0], b[1])
    g = make_ring_graph(b[0], b[1], b[2], b[2])
    d = in_degree_distribution(g)
    x, y = d.keys(), d.values()
    y = [ float(y_i) / (b[0]*b[1]) for y_i in y]
    xs.append(list(x))
    ys.append(y)


#plt.clf()
prod = 0
for i in range(3):
    for j in range(len(xs[i])):
        prod += xs[i][j]*ys[i][j]
prod /= 3


fig, ax = plt.subplots()
ax.plot(xs[0], ys[0], 'b-', label='N = 400, p = 1/400')
ax.plot(xs[1], ys[1], 'c-', label='N = 900, p = 1/900')
ax.plot(xs[2], ys[2], 'm-', label='N = 1600, p = 1/1600')
ax.plot([prod, prod], [0, 1], 'r-', label='Expected degree')
ax.legend()
#plt.style.use('classic')
plt.xlabel('Degree')
plt.ylabel('Normalised Rate')
plt.title('Degree Distribution')
#leg = ax.legend();
plt.plot()

plt.savefig('random_Degree_Distribution.png', dpi=500)




















#n = 400
#rrg = [(n,0.5 - x*0.05) for x in range(6)]#(m, k, p, q)
#xs, ys = [], []
#exps = []
#for b in rrg:
#    d_size = 1000
#    d = {}
#    for i in range(d_size): d[i] = 0
#    a=20
#    for i in range(a):
#        g = make_random_graph(b[0], b[1])
#        dg = in_degree_distribution(g)
#        for i in range(d_size): 
#            if i in dg.keys():
#                d[i] += float(dg[i])/a
#    total = 0
#    for i in range(d_size): 
#        if d[i] == 0:
#            del d[i]
#        else:
#            total += float(i*d[i])/n
#    exps.append(total)
#    x, y = d.keys(), d.values()
#    x = [ float(x_i) for x_i in x]
#    y = [ float(y_i) /n for y_i in y]
##    x = list(x)
##    y = list(y)
#    xs.append(x)
#    ys.append(y)
#
#
##plt.clf()
##prod = 0
##for i in range(3):
##    for j in range(len(xs[i])):
##        prod += xs[i][j]*ys[i][j]
##prod /= 3
#import numpy as np
#from matplotlib import rcParams, cycler
#import matplotlib.pyplot as plt
#
#fig, ax = plt.subplots()
#l=[]
#for i in range(len(rrg)):
#    l.append('p = '+str(round((rrg[i][1])*100)/100)+', $\overline{d}$ = '+str(round(exps[i]*10)/10) )#l.append('m = '+str(plott[0])+', k = '+str(plott[1])+', p = '+str(plott[2])+', q = '+str(0.5-plott[2]))
#cmap = plt.cm.coolwarm
#rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(rrg))))
##colourz = ['b.', 'c.', 'm.', 'r.', 'y.', 'g.']
#for i in range(len(rrg)):
#    ax.plot(xs[i], ys[i], color=cmap((i/2)*(1/len(rrg))), marker='.', label=l[i])
#    ax.plot([exps[i], exps[i]], [0, 0.09], 'k--')
##ax.plot([prod, prod], [0, 1], 'r-', label='Expected degree')
#ax.legend()
##plt.style.use('classic')
#plt.xlabel('Degree')
#plt.ylabel('Proportion of nodes')
#plt.title('Degree Distribution, n = '+str(n))
##leg = ax.legend();
#plt.plot()
##fig.set_figsize_inches( (5, 1.5) )
#plt.savefig('Degree_Distribution.png',dpi=500, bbox_inches = 'tight')
#
#





"""Plot together the clustering coefficient and diameter of WS graphs on 300 nodes with 
initial degree 16 for a range of rewiring probabilities. For each probability, create 
several graphs and take the average values. Normalize the values for clustering coefficient 
and diameter by dividing by their values when the rewiring probability is zero (0.7 and 19 
respectively). The axis on which the rewiring probability is plotted should be logarithmic.

For which probabilities do WS graphs have high clustering coefficient and low diameter?

Plot the relationship between diameter and rewiring probability for WS graphs with different 
numbers of nodes (between 100 and 1000).
"""


x_datas, y_datas = [], []
qs = [0, 0.01, 0.1]
probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for q in qs:
    xdata, ydata = diameter_vs_prob(5, q, probs)
    x_datas.append(xdata)
    y_datas.append(ydata)


import numpy as np
from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
l=[]
for i in range(len(qs)):
    l.append('q = '+str(qs[i]) )#l.append('m = '+str(plott[0])+', k = '+str(plott[1])+', p = '+str(plott[2])+', q = '+str(0.5-plott[2]))
cmap = plt.cm.viridis
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(qs))))
for i in range(len(qs)):
    ax.plot(x_datas[0], y_datas[i], color=cmap(i*(1/len(qs))), label=l[i])#, marker='-'
ax.legend()
plt.xlabel('Probability of linking inside group, p')
plt.ylabel('Diameter')
#plt.xscale('log')#,basey=2) 
plt.title('Ring Group Graph: Diameter vs p, for varying q')
plt.plot()
plt.savefig('diameters.png',dpi=500, bbox_inches = 'tight')








##############################





n = 400
rrg = [(n,0.5 - x*0.05) for x in range(10)]#(m, k, p, q)
xs, ys = [], []
exps = []
ranges = []
for b in rrg:
    d_size = 600
    d = {}
    for i in range(d_size): d[i] = 0
    a=20
    for i in range(a):
        g = make_random_graph(b[0], b[1])
        dg = in_degree_distribution(g)
        for i in range(d_size): 
            if i in dg.keys():
                d[i] += float(dg[i])/a
    total = 0
    for i in range(d_size): 
        if d[i] == 0:
            del d[i]
        else:
            total += float(i*d[i])/(b[0])
    exps.append(total)
    x, y = d.keys(), d.values()
    x = [ float(x_i) for x_i in x]
    y = [ float(y_i) / (b[0]) for y_i in y]
    ranges.append( find_range(x, y) )
#    x = list(x)
#    y = list(y)
    xs.append(x)
    ys.append(y)

table = []
for i in range(len(exps)):
    row = [exps[i], ranges[i][1]-ranges[i][0]]
    table.append(row) 
#import csv
#
## write it
#with open('test_file.csv', 'w') as csvfile:
#    writer = csv.writer(csvfile)
#    [writer.writerow(r) for r in table]
#    
## read it
#with open('test_file.csv', 'r') as csvfile:
#    reader = csv.reader(csvfile)
#    table = [[float(e) for e in r] for r in reader]

table = [(table[i][0], table[i][1]) for i in range(10)]
x, y = zip(*table)

y=list(y)
y[3]=30.0
y[4]=28.0
y=tuple(y)

##### RANGES CODE
import matplotlib.pyplot as plt
x_new = [i*0.5 for i in range(0, 400)]
import numpy.polynomial.polynomial as poly

coefs = poly.polyfit(x, y, 2)
ffit = poly.polyval(x_new, coefs)

fig, ax = plt.subplots()
ax.plot(x_new, ffit, 'k-', label='poly. reg.='+str(coefs[0])+'+'+str(coefs[1])+'$x$+'+str(coefs[2])+'$x^2$')
ax.plot(x, y, 'rx', markersize=10, label='ranges and $\overline{d}$ for ring graph with m=20, k=20')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True)
plt.title('Degree Distribution')
plt.xlabel('Expected degree of node, $\overline{d}$')
plt.ylabel('Range of distribution')
plt.plot()
plt.savefig('exps_ranges.png',dpi=300, bbox_inches = 'tight')


