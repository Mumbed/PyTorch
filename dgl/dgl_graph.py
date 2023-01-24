# import dgl
# import torch as th
#
# u,v = th.tensor([0,0,0,1]), th.tensor([1,2,3,3])
# g = dgl.graph((u,v))
#
# print(g)
#
# print(g.nodes())
#
# print(g.edges())
#
# print(g.edges(form='all'))
#
# g = dgl.graph((u,v), num_nodes=8)
#
# #bidirected graph
# bg = dgl.to_bidirected(g)
# bg.edges()
import dgl
import torch as th

g = dgl.graph(([0,0,1,5],[1,2,2,0]))
print(g)
print(g.nodes())
print(g.edges())

g.ndata['x'] = th.ones(g.num_nodes(),3)
g.edata['y'] = th.ones(g.num_edges(), dtype=th.int32)
print(g)