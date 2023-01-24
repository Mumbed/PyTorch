import dgl
import torch as th

graph_data = {
    ('drug', 'interacts', 'drug'): (th.tensor([0,1]), th.tensor([1,2])),
    ('drug', 'interacts', 'gene'): (th.tensor([0,1]), th.tensor([2,3])),
    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2])),
}
g = dgl.heterograph(graph_data)
#print(g.ntypes)
#print(g.etypes)

#print(g.canonical_etypes)

print(g.num_nodes())
print(g.num_nodes('drug'))
gx = dgl.to_networks(g)
gx.draw_networks(gx)