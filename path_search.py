from copy import deepcopy
import networkx as nx

class Path(object):
    def __init__(self, init_nodes=None, init_dirs=None):
        if init_nodes is not None:
            self.nodes = deepcopy(init_nodes)
        else:
            self.nodes = []

        if init_dirs is not None:
            self.path_dirs = deepcopy(init_dirs)
        else:
            self.path_dirs = []

    def AddNode(self, new_node, new_dir):
        self.nodes.append(new_node)
        self.path_dirs.append(new_dir)

    def ExtendPath(self, new_path, con_dir):
        self.nodes.extend(new_path.nodes)
        self.path_dirs.extend([con_dir] + new_path.path_dirs)

    def IfCausalPath(self):
        for idx in range(len(self.path_dirs)):
            if self.path_dirs[idx] != 1:
                    return False
        return True

    def IfPosActive(self):
        non_zero_dirs = [dir for dir in self.path_dirs if dir != 0]
        if len(non_zero_dirs) > 1:
            for i in range(len(non_zero_dirs) - 1):
                if non_zero_dirs[i] == 1 and non_zero_dirs[i + 1] == -1:
                    return False
        return True

    def IfConPosActive(self, C):
        for idx in range(len(self.path_dirs) - 1):
            if self.path_dirs[idx] == 1 and self.path_dirs[idx + 1] == -1:
                if self.nodes[idx + 1] not in C:
                    return False
        return True


def UpdateRelation(pos_active_paths):
    adj_dict = {}
    dir_succ_dict = {}
    dir_pre_dict = {}
    for pos_active_path in pos_active_paths:
        path_nodes = pos_active_path.nodes
        for i in range(len(path_nodes) - 1):
            if path_nodes[i] not in adj_dict:
                adj_dict[path_nodes[i]] = set()
                dir_succ_dict[path_nodes[i]] = set()
                dir_pre_dict[path_nodes[i]] = set()
            if path_nodes[i + 1] not in adj_dict:
                adj_dict[path_nodes[i + 1]] = set()
                dir_succ_dict[path_nodes[i + 1]] = set()
                dir_pre_dict[path_nodes[i + 1]] = set()
            adj_dict[path_nodes[i]].add(path_nodes[i + 1])
            adj_dict[path_nodes[i + 1]].add(path_nodes[i])
            dir_succ_dict[path_nodes[i]].add(path_nodes[i + 1])
            dir_pre_dict[path_nodes[i + 1]].add(path_nodes[i])

    for pos_active_path in pos_active_paths:
        path_nodes = pos_active_path.nodes
        for i in range(len(path_nodes) - 1):
            for j in range(i + 1, len(path_nodes)):
                if path_nodes[j] in dir_pre_dict[path_nodes[i]]:
                    dir_pre_dict[path_nodes[i]].remove(path_nodes[j])
                if path_nodes[i] in dir_succ_dict[path_nodes[j]]:
                    dir_succ_dict[path_nodes[j]].remove(path_nodes[i])

    return adj_dict, dir_pre_dict, dir_succ_dict


def UpdateRelationBuckets(pos_active_paths, bucket_map):
    adj_dict = {}
    dir_succ_dict = {}
    dir_pre_dict = {}
    for pos_active_path in pos_active_paths:
        path_nodes = pos_active_path.nodes
        for i in range(len(path_nodes) - 1):
            b_i = bucket_map[path_nodes[i]]
            b_ip1 = bucket_map[path_nodes[i + 1]]
            if b_i not in adj_dict:
                adj_dict[b_i] = set()
                dir_succ_dict[b_i] = set()
                dir_pre_dict[b_i] = set()
            if b_ip1 not in adj_dict:
                adj_dict[b_ip1] = set()
                dir_succ_dict[b_ip1] = set()
                dir_pre_dict[b_ip1] = set()
            if b_i == b_ip1:
                continue
            adj_dict[b_i].add(b_ip1)
            adj_dict[b_ip1].add(b_i)
            dir_succ_dict[b_i].add(b_ip1)
            dir_pre_dict[b_ip1].add(b_i)

    for pos_active_path in pos_active_paths:
        path_nodes = pos_active_path.nodes
        for i in range(len(path_nodes) - 1):
            b_i = bucket_map[path_nodes[i]]
            for j in range(i + 1, len(path_nodes)):
                b_j = bucket_map[path_nodes[j]]
                if b_i == b_j:
                    continue
                if b_j in dir_pre_dict[b_i]:
                    dir_pre_dict[b_i].remove(b_j)
                if b_i in dir_succ_dict[b_j]:
                    dir_succ_dict[b_j].remove(b_i)

    return adj_dict, dir_pre_dict, dir_succ_dict


def SearchNodePermutation(cur_seq, f_nodes, dir_succ_dict):
    if len(f_nodes) < 2:
        return [cur_seq + f_nodes]
    return_seq = []
    for f_node in f_nodes:
        if_valid = True
        for f_node_other in f_nodes:
            if f_node != f_node_other:
                if f_node in dir_succ_dict[f_node_other]:
                    if_valid = False
                    break
        if if_valid:
            in_seq = deepcopy(cur_seq)
            in_seq.append(f_node)
            in_f_nodes = deepcopy(f_nodes)
            in_f_nodes.remove(f_node)
            return_seq += SearchNodePermutation(in_seq, in_f_nodes, dir_succ_dict)

    return return_seq


def SearchPathPermutation(G, A_name):
    pos_active_paths = []
    tmp_path = Path([A_name], None)
    search_queues = [tmp_path]
    while len(search_queues) > 0:
        cur_path = search_queues[0]
        del (search_queues[0])
        cur_last_node = cur_path.nodes[-1]

        if len(cur_path.nodes) > 1:
            pos_active_paths.append(cur_path)

        for n in G.adj[cur_last_node]:
            if n not in cur_path.nodes:
                if G.edges[cur_last_node, n]["From"] is None:
                    new_dir = 0
                elif G.edges[cur_last_node, n]["From"] == cur_last_node:
                    new_dir = 1
                else:
                    new_dir = -1

                if len(cur_path.path_dirs) == 0 or cur_path.path_dirs[-1] * new_dir >= 0 or (
                        cur_path.path_dirs[-1] == -1):
                    new_path = Path(cur_path.nodes, cur_path.path_dirs)
                    new_path.AddNode(n, new_dir)
                    if new_path.IfPosActive():
                        search_queues.append(new_path)

    active_features = set()
    for pos_active_path in pos_active_paths:
        for path_node in pos_active_path.nodes:
            active_features.add(path_node)

    bucket_map = {path_node: path_node for path_node in active_features}

    while True:
        adj_dict, dir_pre_dict, dir_succ_dict = UpdateRelationBuckets(pos_active_paths, bucket_map)

        search_nodes = list(deepcopy(dir_succ_dict[A_name]))
        active_nodes = [node for node in adj_dict.keys()]
        query_nodes = deepcopy(active_nodes)
        query_nodes.remove(A_name)
        finish_nodes = []
        while len(search_nodes) > 0:
            cur_node = search_nodes[0]
            if not adj_dict[cur_node].issubset(dir_succ_dict[cur_node].union(dir_pre_dict[cur_node])):
                diff_set = adj_dict[cur_node].difference(dir_succ_dict[cur_node].union(dir_pre_dict[cur_node]))
                # for item in diff_set:
                new_bucket_nodes = sorted([cur_node] + list(diff_set))
                new_bucket = ":".join(new_bucket_nodes)
                for key in bucket_map:
                    if bucket_map[key] == cur_node or bucket_map[key] in diff_set:
                        bucket_map[key] = new_bucket
                break
            else:
                query_nodes.remove(cur_node)
                finish_nodes.append(cur_node)
                search_nodes.remove(cur_node)
                for tmp_node in dir_succ_dict[cur_node]:
                    if tmp_node not in finish_nodes and tmp_node not in search_nodes:
                        search_nodes.append(tmp_node)

        if len(query_nodes) == 0:
            break

    active_nodes = [node for node in adj_dict.keys()]
    active_paths_b = []
    search_queues = [[A_name]]
    while len(search_queues) > 0:
        cur_path = search_queues[0]
        del (search_queues[0])
        cur_last_node = cur_path[-1]

        if len(cur_path) > 1:
            # print(cur_path)
            active_paths_b.append(cur_path)

        for n in dir_succ_dict[cur_last_node]:
            new_path = deepcopy(cur_path)
            new_path.append(n)
            search_queues.append(new_path)

    active_paths_dict = {}
    for active_path in active_paths_b:
        if active_path[-1] not in active_paths_dict:
            active_paths_dict[active_path[-1]] = []
        active_paths_dict[active_path[-1]].append(active_path)

    all_node_permutations = SearchNodePermutation([], active_nodes, dir_succ_dict)
    all_edge_permutations, all_path_permutations, all_cut_map = [], [], []
    for node_permutation in all_node_permutations:
        edge_permutation = []
        path_permutation = []
        cut_map = [0]
        path_order = {}
        tmp_node_order = {node_permutation[i]: i for i in range(len(node_permutation))}
        for tmp_node in node_permutation[1:]:
            tmp_pre = dir_pre_dict[tmp_node]
            sorted_pre = sorted(tmp_pre, key=lambda pre: tmp_node_order[pre])
            for pre in sorted_pre:
                edge_permutation.append((pre, tmp_node))
                tmp_paths = [t_path for t_path in active_paths_dict[tmp_node] if t_path[-2] == pre]
                if len(tmp_paths) == 1:
                    path_permutation.append(tmp_paths[0])
                    path_order[tuple(tmp_paths[0])] = len(path_order)
                else:
                    try:
                        sorted_paths = sorted(tmp_paths, key=lambda path: path_order[tuple(path[:-1])])
                    except Exception:
                        a = 1
                    path_permutation.extend(deepcopy(sorted_paths))
                    for sorted_path in sorted_paths:
                        path_order[tuple(sorted_path)] = len(path_order)
            cut_map.append(len(path_permutation))
        all_edge_permutations.append(edge_permutation)
        all_path_permutations.append(path_permutation)
        all_cut_map.append(cut_map)
    return active_nodes, dir_pre_dict, all_node_permutations, all_edge_permutations, all_path_permutations


def SearchCausalPath(G, A_name, edge_list, f_names):
    causal_paths = []
    tmp_path = Path([A_name], None)
    search_queues = [tmp_path]
    while len(search_queues) > 0:
        cur_path = search_queues[0]
        del (search_queues[0])
        cur_last_node = cur_path.nodes[-1]

        if len(cur_path.nodes) > 1:
            causal_paths.append(cur_path)

        for n in G.adj[cur_last_node]:
            if n not in cur_path.nodes:
                if G.edges[cur_last_node, n]["From"] is None:
                    new_dir = 0
                elif G.edges[cur_last_node, n]["From"] == cur_last_node:
                    new_dir = 1
                else:
                    new_dir = -1

                if len(cur_path.path_dirs) == 0 or cur_path.path_dirs[-1] * new_dir >= 0 or (
                        cur_path.path_dirs[-1] == -1):
                    new_path = Path(cur_path.nodes, cur_path.path_dirs)
                    new_path.AddNode(n, new_dir)
                    if new_path.IfCausalPath():
                        search_queues.append(new_path)

    parent_dict = {node:[] for node in f_names}
    child_dict = {node: [] for node in f_names}
    for edge in edge_list:
        if edge[2]['From'] is not None:
            if edge[2]['From'] == edge[0]:
                parent_dict[edge[1]].append(edge[0])
                child_dict[edge[0]].append(edge[1])
            elif edge[2]['From'] == edge[1]:
                parent_dict[edge[0]].append(edge[1])
                child_dict[edge[1]].append(edge[0])

    active_nodes = [causal_path.nodes[-1] for causal_path in causal_paths]
    active_nodes = list(set(active_nodes))
    path_list = [causal_path.nodes for causal_path in causal_paths]
    node_permutations = SearchNodePermutation([], active_nodes, child_dict)

    return active_nodes, parent_dict, node_permutations, path_list


def SearchPathPermutationConditional(G, A_name, y_name):
    pos_active_paths = []
    tmp_path = Path([A_name], None)
    search_queues = [tmp_path]
    while len(search_queues) > 0:
        cur_path = search_queues[0]
        del (search_queues[0])
        cur_last_node = cur_path.nodes[-1]

        if len(cur_path.nodes) > 1:
            pos_active_paths.append(cur_path)

        for n in G.adj[cur_last_node]:
            if n not in cur_path.nodes:
                if G.edges[cur_last_node, n]["From"] is None:
                    new_dir = 0
                elif G.edges[cur_last_node, n]["From"] == cur_last_node:
                    new_dir = 1
                else:
                    new_dir = -1

                if len(cur_path.path_dirs) == 0 or (cur_path.path_dirs[-1] * new_dir < 0 and cur_path.nodes[-1] == y_name ) or cur_path.path_dirs[-1] * new_dir >= 0 or (cur_path.path_dirs[-1] == -1):
                    new_path = Path(cur_path.nodes, cur_path.path_dirs)
                    new_path.AddNode(n, new_dir)
                    if new_path.IfConPosActive([y_name]):
                        search_queues.append(new_path)

    active_features = set()
    for pos_active_path in pos_active_paths:
        for path_node in pos_active_path.nodes:
            active_features.add(path_node)

    for (path_idx, pos_active_path) in enumerate(pos_active_paths):
        if len(pos_active_path.nodes) and pos_active_path.nodes[-1] == y_name:
            del pos_active_paths[path_idx]

    for pos_active_path in pos_active_paths:
        for (node_idx, path_node) in enumerate(pos_active_path.nodes):
            if path_node == y_name:
                del pos_active_path.nodes[node_idx]
                del pos_active_path.path_dirs[node_idx - 1]

    bucket_map = {path_node: path_node for path_node in active_features}

    while True:
        adj_dict, dir_pre_dict, dir_succ_dict = UpdateRelationBuckets(pos_active_paths, bucket_map)

        search_nodes = list(deepcopy(dir_succ_dict[A_name]))
        active_nodes = [node for node in adj_dict.keys()]
        query_nodes = deepcopy(active_nodes)
        query_nodes.remove(A_name)
        finish_nodes = []
        while len(search_nodes) > 0:
            cur_node = search_nodes[0]
            if not adj_dict[cur_node].issubset(dir_succ_dict[cur_node].union(dir_pre_dict[cur_node])):
                diff_set = adj_dict[cur_node].difference(dir_succ_dict[cur_node].union(dir_pre_dict[cur_node]))
                # for item in diff_set:
                new_bucket_nodes = sorted([cur_node] + list(diff_set))
                new_bucket = ":".join(new_bucket_nodes)
                for key in bucket_map:
                    if bucket_map[key] == cur_node or bucket_map[key] in diff_set:
                        bucket_map[key] = new_bucket
                break
            else:
                query_nodes.remove(cur_node)
                finish_nodes.append(cur_node)
                search_nodes.remove(cur_node)
                for tmp_node in dir_succ_dict[cur_node]:
                    if tmp_node not in finish_nodes and tmp_node not in search_nodes:
                        search_nodes.append(tmp_node)

        if len(query_nodes) == 0:
            break

    active_nodes = [node for node in adj_dict.keys()]
    active_paths_b = []
    search_queues = [[A_name]]
    while len(search_queues) > 0:
        cur_path = search_queues[0]
        del (search_queues[0])
        cur_last_node = cur_path[-1]

        if len(cur_path) > 1:
            # print(cur_path)
            active_paths_b.append(cur_path)

        for n in dir_succ_dict[cur_last_node]:
            new_path = deepcopy(cur_path)
            new_path.append(n)
            search_queues.append(new_path)

    active_paths_dict = {}
    for active_path in active_paths_b:
        if active_path[-1] not in active_paths_dict:
            active_paths_dict[active_path[-1]] = []
        active_paths_dict[active_path[-1]].append(active_path)

    all_node_permutations = SearchNodePermutation([], active_nodes, dir_succ_dict)
    all_edge_permutations, all_path_permutations, all_cut_map = [], [], []
    for node_permutation in all_node_permutations:
        edge_permutation = []
        path_permutation = []
        cut_map = [0]
        path_order = {}
        tmp_node_order = {node_permutation[i]: i for i in range(len(node_permutation))}
        for tmp_node in node_permutation[1:]:
            tmp_pre = dir_pre_dict[tmp_node]
            sorted_pre = sorted(tmp_pre, key=lambda pre: tmp_node_order[pre])
            for pre in sorted_pre:
                edge_permutation.append((pre, tmp_node))
                tmp_paths = [t_path for t_path in active_paths_dict[tmp_node] if t_path[-2] == pre]
                if len(tmp_paths) == 1:
                    path_permutation.append(tmp_paths[0])
                    path_order[tuple(tmp_paths[0])] = len(path_order)
                else:
                    try:
                        sorted_paths = sorted(tmp_paths, key=lambda path: path_order[tuple(path[:-1])])
                    except Exception:
                        a = 1
                    path_permutation.extend(deepcopy(sorted_paths))
                    for sorted_path in sorted_paths:
                        path_order[tuple(sorted_path)] = len(path_order)
            cut_map.append(len(path_permutation))
        all_edge_permutations.append(edge_permutation)
        all_path_permutations.append(path_permutation)
        all_cut_map.append(cut_map)
    return active_nodes, dir_pre_dict, all_node_permutations, all_edge_permutations, all_path_permutations


if __name__ == "__main__":
    edge_list = []
    f_names = set()
    with open("data/adult_edge_eo") as fout:
        lines = fout.readlines()
        for line in lines:
            line_info = line.strip().split(",")
            if line_info[2] == "none":
                edge = (line_info[0], line_info[1], {'From': None})
            else:
                edge = (line_info[0], line_info[1], {'From': line_info[2]})
            edge_list.append(edge)
            f_names.add(line_info[0])
            f_names.add(line_info[1])
    G = nx.Graph()
    G.add_nodes_from(f_names)
    G.add_edges_from(edge_list)
    parent_dict = {node:[] for node in f_names}
    for edge in edge_list:
        if edge[2]['From'] is not None:
            if edge[2]['From'] == edge[0]:
                parent_dict[edge[1]].append(edge[0])
            else:
                parent_dict[edge[0]].append(edge[1])

    ancestor_dict = {}
    for node in f_names:
        search_queue = deepcopy(parent_dict[node])
        ancestor_dict[node] = []
        while len(search_queue) > 0:
            cur_node = search_queue[0]
            search_queue.remove(cur_node)
            ancestor_dict[node].append(cur_node)
            for p_node in parent_dict[cur_node]:
                if p_node not in ancestor_dict[node] and p_node not in search_queue:
                    search_queue.append(p_node)

    SearchPathPermutationConditional(G, "Sex", "Outcome")