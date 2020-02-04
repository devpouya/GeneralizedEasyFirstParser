import numpy as np
import torch.nn.functional as F
import torch
from utils import constants


def find_cycle(parents, curr_nodes):
    length = len(parents)
    added = np.zeros([length], np.bool)
    added[0] = True
    cycle = set()
    findcycle = False
    for i in range(1, length):
        if added[i] or not curr_nodes[i]:
            continue
        # init cycle
        tmp_cycle = set()
        tmp_cycle.add(i)
        added[i] = True
        findcycle = True
        node = i

        while parents[node] not in tmp_cycle:
            node = parents[node]
            if added[node]:
                findcycle = False
                break
            added[node] = True
            tmp_cycle.add(node)

        if findcycle:
            orig_node = node
            cycle.add(orig_node)
            node = parents[orig_node]
            while node != orig_node:
                cycle.add(node)
                node = parents[node]
            break

    return cycle if findcycle else set()


def remove_cycle(scores, curr_nodes, incoming, outgoing, reached, parents, cycle):
    # pylint: disable=too-many-locals
    cycle = np.array(list(cycle))
    cyc_weight = np.sum([scores[parents[node], node] for node in cycle])

    cycle_start = cycle[0]
    for i in range(len(parents)):
        if not curr_nodes[i] or i in cycle:
            continue

        src_node = cycle[np.argmax(scores[cycle, i])]
        cyc_scrs = np.array([scores[i, j] - scores[parents[j], j] for j in cycle])
        tgt_node = cycle[np.argmax(cyc_scrs)]

        scores[cycle_start, i] = scores[src_node, i]
        incoming[cycle_start, i] = incoming[src_node, i]
        outgoing[cycle_start, i] = outgoing[src_node, i]
        scores[i, cycle_start] = cyc_weight + np.max(cyc_scrs)
        outgoing[i, cycle_start] = outgoing[i, tgt_node]
        incoming[i, cycle_start] = incoming[i, tgt_node]

    cycle_reached = []
    for i, cyc_node in enumerate(cycle):
        cycle_reached.append(set())
        for cyc_reached in reached[cyc_node]:
            cycle_reached[i].add(cyc_reached)

    for cyc_node in cycle[1:]:
        curr_nodes[cyc_node] = False
        for cyc_reached in reached[cyc_node]:
            reached[cycle[0]].add(cyc_reached)
    return cycle_reached


def chu_liu_edmonds(scores, curr_nodes, incoming, outgoing, nodes_reached, final_edges):
    length = len(curr_nodes)
    parents = np.zeros([length], dtype=np.int32)
    # create best graph
    parents[0] = -1
    for i in range(1, length):
        # only interested at current nodes
        if curr_nodes[i]:
            parents[i] = np.argmax(scores[:, i])

    # find a cycle
    cycle = find_cycle(parents, curr_nodes)
    # no cycles, get all edges and return them.
    if len(cycle) == 0:
        final_edges[0] = -1
        for i in range(1, length):
            if not curr_nodes[i]:
                continue
            final_edges[outgoing[parents[i], i]] = incoming[parents[i], i]
        return

    cycle_reached = remove_cycle(scores, curr_nodes, incoming, outgoing,
                                 nodes_reached, parents, cycle)

    chu_liu_edmonds(scores, curr_nodes, incoming, outgoing, nodes_reached, final_edges)

    # check each node in cycle, if one of its reached is a key in the final_edges, it is the one.
    found = False
    final_parent = -1
    for i, node in enumerate(cycle):
        for reached_node in cycle_reached[i]:
            if reached_node in final_edges:
                final_parent = node
                found = True
                break
        if found:
            break

    node = parents[final_parent]
    while node != final_parent:
        final_edges[outgoing[parents[node], node]] = incoming[parents[node], node]
        node = parents[node]


def get_sentence_mst(logprob):
    length = len(logprob)
    parents = np.zeros(length, np.int32)
    incoming = np.zeros([length, length], dtype=np.int32)
    outgoing = np.zeros([length, length], dtype=np.int32)
    curr_nodes = np.ones([length], dtype=np.bool)
    nodes_reached = [{source} for source in range(length)]
    np.fill_diagonal(logprob, 0)  # Remove self edges
    for source in range(length):
        incoming[source, source+1:] = source
        incoming[source+1:, source] = np.arange(source + 1, length)

        outgoing[source, source+1:] = np.arange(source + 1, length)
        outgoing[source+1:, source] = source

    final_edges = dict()
    chu_liu_edmonds(logprob, curr_nodes, incoming, outgoing, nodes_reached, final_edges)

    for child, parent in final_edges.items():
        parents[child] = parent

    parents[0] = 0
    return parents


def get_mst_batch(h_logits, lengths):
    input_shape = h_logits.shape
    batch_size = input_shape[0]
    max_length = input_shape[2]

    heads_tgt = np.zeros([batch_size, max_length], dtype=np.int32)
    for i in range(batch_size):
        length = lengths[i]
        logprob = F.log_softmax(h_logits[i, :length, :length], dim=-1).cpu().numpy()

        logprob = logprob.transpose()
        logprob = logprob - logprob.min() + 1e-6

        heads_tgt[i, :length] = get_sentence_mst(logprob)

    return torch.LongTensor(heads_tgt).to(device=constants.device)
