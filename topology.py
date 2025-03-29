#import plotly.offline as py
from plotly.graph_objs import *
import networkx as nx
import numpy as np

# py.init_notebook_mode()
# %matplotlib inline


def generate_topology(N_node=20, prob=0.25):
    G=nx.random_geometric_graph(N_node, prob)
    pos=nx.get_node_attributes(G,'pos')

    dmin=1
    ncenter=0
    for n in pos:
        x,y = pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d
        
    p=nx.single_source_shortest_path_length(G,ncenter)

    return G



def plot_topology(G, **kwargs):
	edge_trace = Scatter(
	    x=[], 
	    y=[], 
	    line=Line(width=kwargs.get('line_width',2),color='#999'),
	    hoverinfo='none',
	    mode='lines')

	for edge in G.edges():
	    x0, y0 = G.node[edge[0]]['pos']
	    x1, y1 = G.node[edge[1]]['pos']
	    edge_trace['x'] += [x0, x1, None]
	    edge_trace['y'] += [y0, y1, None]

	node_trace = Scatter(
	    x=[], 
	    y=[], 
	    #text=[],
	    mode='markers+text', 
	    hoverinfo='text',
	    text = [str(i) for i in range(100)],
	    textposition='middle center', 
	    #(enumerated: "top left" | "top center" | "top right" | "middle left" | "middle center" | "middle right" | "bottom left" | "bottom center" | "bottom right" ) 
	    textfont={"size": 18, "family": "Arial", 'color': "#000"},
	    marker=Marker(
	        showscale=False,
	        # colorscale options
	        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
	        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
	        colorscale='YIGnBu',
	        reversescale=True,
	        color=[], 
	        size=kwargs.get('size', 20),         
	        #         colorbar=dict(
	        #             thickness=15,
	        #             title='Node Connections',
	        #             xanchor='left',
	        #             titleside='right'
	        #         ),
	        line=dict(width=2)
	        )
	    )

	for node in G.nodes():
	    x, y = G.node[node]['pos']
	    node_trace['x'].append(x)
	    node_trace['y'].append(y)


	for node, adjacencies in enumerate(G.adjacency_list()):
	    node_trace['marker']['color'].append(len(adjacencies))
	    node_info = '# of connections: '+str(len(adjacencies))
	    # node_trace['text'].append(node_info)

	fig = Figure(data=Data([edge_trace, node_trace]),
	             layout=Layout(
	                # title='<br>Network Topology',
	                # titlefont=dict(size=16),
	                showlegend=False, 
	                width=kwargs.get('fig_width', 600),
	                height=kwargs.get('fig_height', 400),
	                hovermode='closest',
	                margin=dict(b=20,l=5,r=5,t=40),
	                annotations=[ dict(
	                    text="",
	                    showarrow=False,
	                    xref="paper", yref="paper",
	                    x=0.005, y=-0.002 ) ],
	                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
	                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

	fig=py.iplot(fig, filename='networkx')



def generate_network(N_node=20, method='average_rule', plot=True, **kwargs):
	'''
		Wrap function to generate topology and plot it. 
		Then by using avg or metropolis rule to generate the combination matrix
		current support key words:   prob----------connected probability

		method only support 'average_rule' and 'metropolis_rule' now
	'''
	
	# start with 1 because assuming every node has self-loop
	indegree = np.ones((N_node,1))
	G = generate_topology(N_node, kwargs.get('prob', 0.25))
	for edge in G.edges():
	    indegree[edge[0]] += 1
	    indegree[edge[1]] += 1

	def avg_rule(G, indegree):
		N_node = indegree.shape[0]
		A = np.zeros((N_node,N_node))
		for e1, e2 in G.edges():
	  		A[e1,e2] = 1./indegree[e2]
	  		A[e2,e1] = 1./indegree[e1]

		for i in range(N_node):
	 		A[i,i] = 1. - np.sum(A[:,i])
		return A

	def metropolis_rule(G, indegree):
		N_node = indegree.shape[0]
		A = np.zeros((N_node,N_node))
		for e1, e2 in G.edges():
	  		A[e1,e2] = 1./max(indegree[e1], indegree[e2])
	  		A[e2,e1] = 1./max(indegree[e1], indegree[e2])

		for i in range(N_node):
	 		A[i,i] = 1. - np.sum(A[:,i])
		return A 

	if plot == True:
		plot_topology(G, **kwargs)

	option = {'average_rule': avg_rule,
			  'metropolis_rule': metropolis_rule}

	if method not in option:
		print ('Currently, only support "average_rule" and "metropolis_rule"')

	return option[method](G, indegree), G

# if __name__ == '__main__':
# 	generate_network()