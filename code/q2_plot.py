from random_ring_graph import make_ring_graph
from q2_code import make_PA_Graph, brilliance, load_graph, count_edges


import matplotlib.pyplot as plt
#import numpy as np
#import scipy.stats as stats 

#b = {}
#for i in pa_graph:
#    b[i] = list( pa_graph[i] )

#brils = []
#degrees = []
#for i in coauth_graph:
#    degrees.append(len(coauth_graph[i]))
#    brils.append(brr[i])
def plot_brilliance(name, tt, dict_nodes_brilliance):
    brills, frequency = xy_brilliance(dict_nodes_brilliance)
    plt.clf()
    #plot degree distribution 
    plt.xlabel('Brilliance')
    plt.ylabel('Normalized Rate')
    plt.title(tt)
    plt.plot(brills, frequency, marker='.', linestyle='None', color='b', dpi = 400)
    plt.savefig(name)


plot_brilliance('coauth_brilliance_rates.png', 'Brilliance Distribution of Citation Graph', brr_dict)

def xy_brilliance(dict_nodes_brilliance):
    brilliances = {}
    for node in dict_nodes_brilliance:
        if dict_nodes_brilliance[node] in brilliances:
            brilliances[dict_nodes_brilliance[node]] += 1
        else:
            brilliances[dict_nodes_brilliance[node]] = 1
    brills, frequency = list(brilliances.keys()), list(brilliances.values())
    return brills, frequency



xss, yss = [], []
ods = [76, 78, 80]

#for od in ods:
#     pa_graph = make_PA_Graph(1560, od)
#     c = count_edges(pa_graph)
#     print('edges = '+str(c))

edges_c = []
k = 5
n = 1560
for od in ods:
    d = {}
    edges_i = []
    for i in range(k):
        pa_graph = make_PA_Graph(n, od)
        edges_i.append(count_edges(pa_graph))
        pabrr_dict = brilliance(pa_graph)
    #    plot_brilliance('pa_brilliance_rates_od='+str(od)+'.png', 'Brilliance Dist. of Preferential Attachment Graph, out degree = '+str(od), pabrr_dict)
    
    
        xs, ys = xy_brilliance(pabrr_dict)
        for i in range(len(xs)):
            if xs[i] in d:
                d[xs[i]] += ys[i]
            else:
                d[xs[i]] = ys[i]
    edges_c.append( float(sum(edges_i))/len(edges_i) )
    xs, ys = d.keys(), d.values()
#    xs = [float(x) for x in xs ]
    ys = [float(y)/k for y in ys ]   
    xss.append(xs)
    yss.append(ys)

import numpy as np
from matplotlib import rcParams, cycler
#import matplotlib.pyplot as plt
plt.clf()

fig, ax = plt.subplots()
l=[]
for i in range(len(ods)):
    l.append('out deg. = '+str(ods[i])+', edges = '+str(round(edges_c[i]) ))
cmap = plt.cm.viridis
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(ods))))
for i in range(len(ods)):
    ax.plot(xss[i], yss[i], color=cmap((float(i)/2)*(1/len(ods))+0.25), marker='.', label=l[i])
ax.legend(loc='right center', fancybox=True, shadow=True)
plt.xlabel('Brilliance')
plt.ylabel('Number of nodes')
plt.title('Brilliance Dist. of Preferential Attachment Graph, N = '+str(n))
plt.plot()
plt.savefig('pa_brilliance_rates.png',dpi=500, bbox_inches = 'tight')




# m*k == 1500
#mks = [(30, 50)]#mks = [(30, 50), (50, 30)]#[(15, 100), (30, 50), (50, 30), (100, 15)]#[(2, 750), (15, 100), (30, 50), (750, 2)]
#pqs = [(0.05, 0.035)]#pqs = [(0.4, 0.015), (0.1, 0.03)]
#for mk in mks:
#    for pq in pqs:
#         rg_graph = make_ring_graph(mk[0], mk[1], pq[0], pq[1])
#         c = count_edges(rg_graph)
#         print('edges = '+str(c))

mkpqs = [(30, 50, 0.05, 0.035), (30, 50, 0.1, 0.03), (50, 30, 0.1, 0.03)]#, (50, 30, 0.4, 0.015)
edges_c = []
xss, yss = [], []   
for mkpq in mkpqs:
    d = {}
    edges_i = []
    for i in range(k):
        rg_graph = make_ring_graph(mkpq[0], mkpq[1], mkpq[2], mkpq[3])
        edges_i.append(count_edges(rg_graph))
        rgbrr_dict = brilliance(rg_graph)
    #    plot_brilliance('pa_brilliance_rates_od='+str(od)+'.png', 'Brilliance Dist. of Preferential Attachment Graph, out degree = '+str(od), pabrr_dict)
    
    
        xs, ys = xy_brilliance(rgbrr_dict)
        for i in range(len(xs)):
            if xs[i] in d:
                d[xs[i]] += ys[i]
            else:
                d[xs[i]] = ys[i]
    edges_c.append( float(sum(edges_i))/len(edges_i) )
    xs, ys = d.keys(), d.values()
#    xs = [float(x) for x in xs ]
    ys = [float(y)/k for y in ys ]   
    xss.append(xs)
    yss.append(ys)
    
import numpy as np
from matplotlib import rcParams, cycler
#import matplotlib.pyplot as plt
plt.clf()

fig, ax = plt.subplots()
l=[]
for i in range(len(mkpqs)):
    l.append('(m, k, p, q) = ('+str(mkpqs[i][0])+', '+str(mkpqs[i][1])+', '+str(mkpqs[i][2])+', '+str(mkpqs[i][3])+'), edges = '+str(round(edges_c[i]) ))
cmap = plt.cm.viridis
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(mkpqs))))
for i in range(len(mkpqs)):
    ax.plot(xss[i], yss[i], color=cmap(i*(1/len(mkpqs))), marker='.', label=l[i])
ax.legend(loc='right upper', fancybox=True, shadow=True)
plt.xlabel('Brilliance')
plt.ylabel('Number of nodes')
plt.title('Brilliance Dist. of Ring Group Graph, N = 1500')
plt.plot()
plt.savefig('rg_brilliance_rates.png',dpi=500, bbox_inches = 'tight')



#plot_brilliance('coauth_brilliance_rates.png', brr_dict)
#x = np.linspace (0, 100, 200) 
#y1 = 1500*stats.gamma.pdf(x, a=6, scale=3.33333)# scale is 1/beta
#x = 1.5*x
#plt.plot(x, y1, "y-", label=(r'$\alpha=29, \beta=3$')) 
#plt.show()

node_index, coauth_graph = load_graph("coauthorship.txt")
c = count_edges(coauth_graph)
print('edges = '+str(c))
brr_dict = brilliance(coauth_graph)
xs, ys = xy_brilliance(brr_dict)


import numpy as np
from matplotlib import rcParams, cycler
#import matplotlib.pyplot as plt
plt.clf()

#fig, ax = plt.subplots()
#l=[]
#for i in range(len(mkpqs)):
#    l.append('Citation')
cmap = plt.cm.viridis
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 3)))
#ax.plot(xs, ys, color=cmap(0.5), marker='.', label=l[i])
#ax.legend(loc='right upper', fancybox=True, shadow=True)
plt.xlabel('Brilliance')
plt.ylabel('Number of nodes')
plt.title('Brilliance Dist. of Citation Graph, N = '+str(len(coauth_graph)) + ', edges = '+ str(c))
plt.scatter(xs, ys, color=cmap(0.5), s = 8)
plt.savefig('citation_rates.png',dpi=500, bbox_inches = 'tight')







