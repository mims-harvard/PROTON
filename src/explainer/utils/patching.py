import dgl
import torch
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax


def forward_exp(
    self,
    g: dgl.DGLGraph,
    x: torch.Tensor,
    ntype: torch.Tensor,
    etype: torch.Tensor,
    *,
    presorted: bool = False,
    eweight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward computation.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    x : torch.Tensor
        A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
    ntype : torch.Tensor
        An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
    etype : torch.Tensor
        An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
    presorted : bool, optional
        Whether *both* the nodes and the edges of the input graph have been sorted by
        their types. Forward on pre-sorted graph may be faster. Graphs created by
        :func:`~dgl.to_homogeneous` automatically satisfy the condition.
        Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.
    eweight : torch.Tensor, optional
        A 1D tensor of edge weights. Shape: :math:`(|E|,)`.

    Returns
    -------
    torch.Tensor
        New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
    """
    self.presorted = presorted
    if g.is_block:
        x_src = x
        x_dst = x[: g.num_dst_nodes()]
        srcntype = ntype
        dstntype = ntype[: g.num_dst_nodes()]
    else:
        x_src = x
        x_dst = x
        srcntype = ntype
        dstntype = ntype
    with g.local_scope():
        k = self.linear_k(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
        q = self.linear_q(x_dst, dstntype, presorted).view(-1, self.num_heads, self.head_size)
        v = self.linear_v(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
        g.srcdata["k"] = k
        g.dstdata["q"] = q
        g.srcdata["v"] = v
        g.edata["etype"] = etype
        g.apply_edges(self.message)
        g.edata["m"] = g.edata["m"] * edge_softmax(g, g.edata["a"]).unsqueeze(-1)

        if eweight is not None:
            eweight = eweight.view(g.edata["m"].shape[0], 1, 1)
            g.edata["m"] = g.edata["m"] * eweight
        g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))

        h = g.dstdata["h"].view(-1, self.num_heads * self.head_size)
        h = self.drop(self.linear_a(h, dstntype, presorted))
        alpha = torch.sigmoid(self.skip[dstntype]).unsqueeze(-1)
        if x_dst.shape != h.shape:
            h = h * alpha + (x_dst @ self.residual_w) * (1 - alpha)
        else:
            h = h * alpha + x_dst * (1 - alpha)
        if self.use_norm:
            h = self.norm(h)
        return h


def model_forward_exp(self, subgraph: dgl.DGLGraph, eweight: torch.Tensor | None = None) -> torch.Tensor:
    """
    Perform a forward pass of the model. Note that the subgraph must be converted to from a
    heterogeneous graph to homogeneous graph for efficiency.

    Args:
        subgraph (dgl.DGLGraph): Subgraph containing the nodes and edges for the current batch.
        eweight (torch.Tensor, optional): A 1D tensor of edge weights. Shape: :math:`(|E|,)`.
    """
    num_sample = 400000
    if subgraph.num_edges() > num_sample:
        device = subgraph.device
        total_edges = subgraph.num_edges()
        perm = torch.randperm(total_edges, device=device)[:num_sample]
        sampled_subgraph = dgl.edge_subgraph(subgraph, perm, relabel_nodes=False)

        if eweight is not None:
            eweight = eweight[perm]

        subgraph = sampled_subgraph

    global_node_indices = subgraph.ndata["node_index"]

    x = self.emb(global_node_indices)

    x = self.conv1(
        subgraph,
        x,
        subgraph.ndata[dgl.NTYPE],
        subgraph.edata[dgl.ETYPE],
        eweight=eweight,
    )
    x = self.norm1(x)
    x = F.leaky_relu(x)
    x = self.conv2(
        subgraph,
        x,
        subgraph.ndata[dgl.NTYPE],
        subgraph.edata[dgl.ETYPE],
        eweight=eweight,
    )

    if self.num_layers == 3:
        x = self.norm2(x)
        x = F.leaky_relu(x)
        x = self.conv3(
            subgraph,
            x,
            subgraph.ndata[dgl.NTYPE],
            subgraph.edata[dgl.ETYPE],
            eweight=eweight,
        )

    return x
