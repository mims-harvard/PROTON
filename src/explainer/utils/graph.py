import dgl
import torch


def model_khop_in_subgraph(query_kg, k_hops, src_indices, dst_indices, query_edge_type, fanout=None):
    fanout = [-1] * k_hops if fanout is None else [fanout] * k_hops

    query_sampler = dgl.dataloading.ShaDowKHopSampler(fanout)

    query_indices = torch.tensor(src_indices + dst_indices)  # .to(device)
    query_indices = torch.unique(query_indices).unsqueeze(1)

    if query_edge_type not in query_kg.canonical_etypes:
        raise ValueError("Edge type not in knowledge graph.")
    src_type = query_edge_type[0]
    dst_type = query_edge_type[2]

    kg_indices = query_kg.ndata["node_index"]
    query_nodes = {key: torch.where(value == query_indices)[1] for key, value in kg_indices.items()}

    if src_type != dst_type:
        src_set = list(set(src_indices))
        src_srt = list(range(len(src_set)))
        src_map = dict(zip(src_set, src_srt, strict=False))
        src_nodes = torch.tensor([src_map[x] for x in src_indices])

        dst_set = list(set(dst_indices))
        dst_srt = list(range(len(dst_set)))
        dst_map = dict(zip(dst_set, dst_srt, strict=False))
        dst_nodes = torch.tensor([dst_map[x] for x in dst_indices])

    else:
        src_dst_set = list(set(src_indices + dst_indices))
        src_dst_srt = list(range(len(src_dst_set)))
        src_dst_map = dict(zip(src_dst_set, src_dst_srt, strict=False))
        src_nodes = torch.tensor([src_dst_map[x] for x in src_indices])
        dst_nodes = torch.tensor([src_dst_map[x] for x in dst_indices])

    edge_graph_data = {etype: ([], []) for etype in query_kg.canonical_etypes}
    edge_graph_data[query_edge_type] = (src_nodes, dst_nodes)
    query_edge_graph = dgl.heterograph(edge_graph_data)
    assert query_kg.ntypes == query_edge_graph.ntypes
    assert query_kg.canonical_etypes == query_edge_graph.canonical_etypes

    if src_type != dst_type:
        src_map_rev = {y: x for x, y in src_map.items()}
        dst_map_rev = {y: x for x, y in dst_map.items()}
        global_src_nodes = torch.tensor([src_map_rev[x.item()] for x in query_edge_graph.nodes(src_type)])
        global_dst_nodes = torch.tensor([dst_map_rev[x.item()] for x in query_edge_graph.nodes(dst_type)])

        node_index_data = {ntype: torch.empty(0) for ntype in query_kg.ntypes}
        node_index_data[src_type] = global_src_nodes
        node_index_data[dst_type] = global_dst_nodes
        query_edge_graph.ndata["node_index"] = node_index_data

    else:
        src_dst_map_rev = {y: x for x, y in src_dst_map.items()}
        global_src_dst_nodes = torch.tensor([src_dst_map_rev[x.item()] for x in query_edge_graph.nodes(src_type)])

        node_index_data = {ntype: torch.empty(0) for ntype in query_kg.ntypes}
        node_index_data[src_type] = global_src_dst_nodes
        query_edge_graph.ndata["node_index"] = node_index_data

    _, _, query_subgraph = query_sampler.sample(query_kg, query_nodes)

    het_subgraph = query_subgraph

    # Convert heterogeneous graph to homogeneous graph for efficiency
    query_subgraph = dgl.to_homogeneous(query_subgraph, ndata=["node_index"])

    return query_subgraph, query_edge_graph, het_subgraph
