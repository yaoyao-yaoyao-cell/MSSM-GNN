import networkx as nx
import torch
from tqdm import tqdm
import math
import numpy as np
import copy
import gc

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0
def are_graphs_isomorphic(graph1, graph2):
    return nx.is_isomorphic(graph1, graph2)

def all_shortest_paths(G, node_tags=None):
    all_paths = []
    for source in G.g.nodes():
        for target in G.g.nodes():
            if source != target:
                shortest_path = nx.shortest_path(G.g, source=source, target=target)
                subgraph_nodes = set(shortest_path)
                subgraph = G.g.subgraph(subgraph_nodes).copy()
                for node in subgraph.nodes():
                    subgraph.nodes[node].update(G.g.nodes[node])
                if node_tags is not None:
                    for node in subgraph.nodes():
                        if node < len(node_tags):
                            subgraph.nodes[node]['node_tags'] = node_tags[node]
                            subgraphinf = S2VGraph(subgraph, G.label, node_tags)
                            all_paths.append(subgraphinf)
    return all_paths

def floyd_warshall(G):
    graph=G.g
    n = len(graph.nodes())
    nodes = list(graph.nodes())
    cost = {i: {j: {'weight': 0 if i == j else float('inf')} for j in nodes} for i in nodes}
    # Add edges to the cost dictionary with their weights
    for edge in graph.edges(data=True):
        i, j, weight = edge
        cost[i][j]['weight'] = weight.get('weight', 1)
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if cost[i][k]['weight'] + cost[k][j]['weight'] < cost[i][j]['weight']:
                    cost[i][j]['weight'] = cost[i][k]['weight'] + cost[k][j]['weight']
                    cost[i][j]['path'] = nx.shortest_path(graph, source=i, target=j)
    # Create a new graph for the shortest paths
    shortest_paths_graph = nx.Graph()
    # Add edges from the shortest paths to the new graph
    for i in nodes:
        for j in nodes:
            if 'path' in cost[i][j]:
                shortest_path_edges = [(cost[i][j]['path'][k], cost[i][j]['path'][k + 1]) for k in range(len(cost[i][j]['path']) - 1)]
                shortest_paths_graph.add_edges_from(shortest_path_edges)
    return shortest_paths_graph

def perform_mwlsp_graph_kernel_computation(G_floyd1, G_floyd2,c,H):
    kernel_score = 0
    for e1 in G_floyd1.g.edges():
        for e2 in G_floyd2.g.edges():
            kernel_score += compute_k(e1, e2,c,H)
    return kernel_score

def compute_k(e1, e2,c,H):
    # Implement comparison scores of graph-substructures
    sim1 = compute_sim1(e1,e2,c)
    sim2 = compute_sim2(e1,e2,H)
    return sim1 * sim2

def compute_sim1(e1, e2, c):
    # Implement sim1 calculation based on edge lengths
    length_diff = abs(len(e1) - len(e2))
    sim1 = max(0, c - length_diff)
    return sim1
def compute_sim2(e1, e2, H):
    # Implement sim2 calculation based on WL scheme and Mahalanobis distance
    L1, L2 = initialize_labels(e1, e2)
    M = compute_covariance_matrix(L1, L2) 
    M= np.linalg.pinv(M) 
    vertex_similarity = np.zeros((len(e1), len(e2)))
    for h in range(H):
        L1 = propagate_labels(L1, e1)
        L2 = propagate_labels(L2, e2)
    for u in range(len(e1)):
            for v in range(len(e2)):
                mahalanobis_distance = compute_mahalanobis_distance(L1[u], L2[v], M)
                vectorized_exp = np.vectorize(lambda x: np.exp(-0.5 * x))
                vertex_similarity[u, v] += np.sum(vectorized_exp(mahalanobis_distance))
    sim2 = np.sum(vertex_similarity)
    return sim2

def initialize_labels(e1, e2):
    L1 = [e1[0],e1[len(e1)-1]]
    L2 = [e2[0],e2[len(e2)-1]]
    return L1,L2
def propagate_labels(labels, graph):
    # Propagate labels using hash function
    new_labels = labels.copy()

    # Assuming graph is a list of nodes
    for node in graph:
        if 0 <= node < len(labels):
            current_label = labels[node]
            new_label = hash_function(current_label, graph)
            new_labels[node] = new_label

    return new_labels

def hash_function(current_label, neighbors):
    current_label_str = str(current_label)
    neighbors_str = ''.join(map(str, neighbors))
    
    hash_value = hash(current_label_str + neighbors_str)
    return hash_value

def compute_covariance_matrix(L1, L2):
    # Assuming L1 and L2 are lists of labels
    labels = list(set(L1 + L2))  # Extract unique labels
    num_labels = len(labels)
    covariance_matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(num_labels):
            # Calculate covariance between labels i and j
            covariance_matrix[i, j] = covariance(L1.count(labels[i]), L2.count(labels[j]))
    return covariance_matrix

def covariance(count1, count2):
    return count1 * count2

def compute_mahalanobis_distance(u, v, M):
    u_minus_v = np.array(u) - np.array(v)
    
    # Check if M is singular
    if np.linalg.det(M) == 0:
        mahalanobis_distance = float('inf')
    else:
        # Check if M is positive definite
        try:
            np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            # Handle non-positive definite matrix
            mahalanobis_distance = float('inf')
        else:
            # Compute Mahalanobis distance
            mahalanobis_distance = np.sqrt(np.dot(np.dot(u_minus_v, np.linalg.inv(M)), u_minus_v))
    
    return mahalanobis_distance


def all_shortest_paths(G):
    all_paths = []
    node_tags=G.node_tags
    subgraph=floyd_warshall(G)
    for node in subgraph.nodes():
        subgraph.nodes[node].update(G.g.nodes[node])
    if node_tags is not None:
        for node in subgraph.nodes():
           if node < len(node_tags):
              subgraph.nodes[node]['node_tags'] = node_tags[node]
              subgraphinf = S2VGraph(subgraph, G.label, node_tags)
              all_paths.append(subgraphinf)
    return all_paths
def load_data(dataset, degree_as_tag):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        all_paths = []
        g_list_first = g_list
        for i in range(len(g_list_first)):
            shortest_path_graph= floyd_warshall(g_list_first[i])
            node_tags=g_list_first[i].node_tags
            for node in shortest_path_graph.nodes():
                shortest_path_graph.nodes[node].update(g_list_first[i].g.nodes[node])
            if node_tags is not None:
                for node in shortest_path_graph.nodes():
                    if node < len(node_tags):
                       shortest_path_graph.nodes[node]['node_tags'] = node_tags[node]
                       subgraphinf = S2VGraph(shortest_path_graph, g_list_first[i].label, node_tags)            
                       all_paths.append(subgraphinf)
        # Iterate over all combinations
        c=2
        H=3
        for i in range(len(g_list_first)):
                for j in range(i + 1, len(g_list_first)):
                        graph1 = all_paths[i]
                        graph2 = all_paths[j]
                        kernel_score=perform_mwlsp_graph_kernel_computation(graph1,graph2,c,H)
                        
                        # Check for division by zero or NaN
                        if np.max(kernel_score) != 0 and not np.isnan(np.max(kernel_score)):
                            normalized_score = int(3 * kernel_score / np.max(kernel_score))
                        else:
                            # Handle the case where division by zero or NaN occurs
                            # Set a default value or handle it according to your needs
                            normalized_score = 0  # or any other suitable default value
                        if normalized_score > 0:
                            weight = normalized_score  # You can adjust this based on your requirements
                            edge1 = (i, j)
                            edge2 = (j, i)
                            graph1.g.add_edge(*edge1, weight=weight)
                            graph2.g.add_edge(*edge2, weight=weight) 

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

class GenGraph(object):
    def __init__(self, data):
        self.data = data
        self.nodes_labels = data.node_labels
        self.vocab = {}
        self.whole_node_count = {}
        self.weight_vocab = {}
        self.node_count = {}
        self.edge_count = {}
        self.removed_nodes = []
        g = self.gen_components()
        g = self.update_weight(g)
        g = self.add_edge(g)
        self.g_final = self.drop_node(g)
        self.num_cliques = self.g_final.number_of_nodes() - len(self.data.g_list)
        # self.num_cliques = len(self.vocab)
        del g, self.vocab, self.data, self.whole_node_count, self.weight_vocab, self.node_count,self.edge_count
        gc.collect()


    def gen_components(self):
        g_list = self.data.g_list
        h_g = nx.Graph()
        for g in tqdm(range(len(g_list)), desc='Gen Components', unit='graph'):
            nodes_labels = self.nodes_labels[g]
            clique_list = []
            mcb = nx.cycle_basis(g_list[g])
            mcb_tuple = [tuple(ele) for ele in mcb]

            edges = []
            for e in g_list[g].edges():
                count = 0
                for c in mcb_tuple:
                    if e[0] in set(c) and e[1] in set(c):
                        count += 1
                        break
                if count == 0:
                    edges.append(e)
            edges = list(set(edges))

            for e in edges:
                weight = g_list[g].get_edge_data(e[0], e[1])['weight']
                edge = ((nodes_labels[e[0]], nodes_labels[e[1]]), weight)
                clique_id = self.add_to_vocab(edge)
                clique_list.append(clique_id)
                if clique_id not in self.whole_node_count:
                    self.whole_node_count[clique_id] = 1
                else:
                    self.whole_node_count[clique_id] += 1

            for m in mcb_tuple:
                weight = tuple(self.find_ring_weights(m, g_list[g]))
                ring = []
                for i in range(len(m)):
                    ring.append(nodes_labels[m[i]])
                cycle = (tuple(ring), weight)
                cycle_id = self.add_to_vocab(cycle)
                clique_list.append(cycle_id)
                if cycle_id not in self.whole_node_count:
                    self.whole_node_count[cycle_id] = 1
                else:
                    self.whole_node_count[cycle_id] += 1

            for e in clique_list:
                self.add_weight(e, g)

            c_list = tuple(set(clique_list))
            for e in c_list:
                if e not in self.node_count:
                    self.node_count[e] = 1
                else:
                    self.node_count[e] += 1

            for e in c_list:
                h_g.add_edge(g, e + len(g_list), weight=(self.weight_vocab[(g, e)]/(len(edges) + len(mcb_tuple))))

            for e in range(len(edges)):
                for i in range(e+1, len(edges)):
                    for j in edges[e]:
                        if j in edges[i]:
                            weight = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge = ((nodes_labels[edges[e][0]], nodes_labels[edges[e][1]]), weight)
                            weight_i = g_list[g].get_edge_data(edges[i][0], edges[i][1])['weight']
                            edge_i = ((nodes_labels[edges[i][0]], nodes_labels[edges[i][1]]), weight_i)
                            final_edge = tuple(sorted((self.add_to_vocab(edge), self.add_to_vocab(edge_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1
            for m in range(len(mcb_tuple)):
                for i in range(m+1, len(mcb_tuple)):
                    for j in mcb_tuple[m]:
                        if j in mcb_tuple[i]:
                            weight = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring = []
                            for t in range(len(mcb_tuple[m])):
                                ring.append(nodes_labels[mcb_tuple[m][t]])
                            cycle = (tuple(ring), weight)

                            weight_i = tuple(self.find_ring_weights(mcb_tuple[i], g_list[g]))
                            ring_i = []
                            for t in range(len(mcb_tuple[i])):
                                ring_i.append(nodes_labels[mcb_tuple[i][t]])
                            cycle_i = (tuple(ring_i), weight_i)

                            final_edge = tuple(sorted((self.add_to_vocab(cycle), self.add_to_vocab(cycle_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1
            for e in range(len(edges)):
                for m in range(len(mcb_tuple)):
                    for i in edges[e]:
                        if i in mcb_tuple[m]:
                            weight_e = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge_e = ((nodes_labels[edges[e][0]], nodes_labels[edges[e][1]]), weight_e)
                            weight_m = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring_m = []
                            for t in range(len(mcb_tuple[m])):
                                ring_m.append(nodes_labels[mcb_tuple[m][t]])
                            cycle_m = (tuple(ring_m), weight_m)

                            final_edge = tuple(sorted((self.add_to_vocab(edge_e), self.add_to_vocab(cycle_m))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1
        return h_g

    def add_to_vocab(self, clique):
        c = copy.deepcopy(clique[0])
        weight = copy.deepcopy(clique[1])
        if len(list(c)) == 2:
            for i in range(len(list(c))):
                if (c, weight) in self.vocab:
                    return self.vocab[(c, weight)]
                else:
                    c = self.shift_right(c)
        else:
            for i in range(len(c)):
                if (c, weight) in self.vocab:
                    return self.vocab[(c, weight)]
                else:
                    c = self.shift_right(c)
                    weight = self.shift_right(weight)
        self.vocab[(c, weight)] = len(list(self.vocab.keys()))
        return self.vocab[(c, weight)]

    def add_weight(self, node_id, g):
        if (g, node_id) not in self.weight_vocab:
            self.weight_vocab[(g, node_id)] = 1
        else:
            self.weight_vocab[(g, node_id)] += 1

    def update_weight(self, g):
        for (u, v) in g.edges():
            if u < len(self.data.g_list):
                g[u][v]['weight'] = g[u][v]['weight'] * (math.log((len(self.data.g_list) + 1) / (self.node_count[v - len(self.data.g_list)] + 1) + 1))
            else:
                g[u][v]['weight'] = g[u][v]['weight'] * (
                    math.log((len(self.data.g_list) + 1) / (self.node_count[u - len(self.data.g_list)] + 1) + 1))
        return g

    def add_edge(self, g):
        edges = list(self.edge_count.keys())
        for i in edges:
            g.add_edge(i[0] + len(self.data.g_list), i[1] + len(self.data.g_list), weight=math.exp(math.log(self.edge_count[i] / math.sqrt(self.whole_node_count[i[0]] * self.whole_node_count[i[1]]))))
        return g

    def drop_node(self, g):
        rank_list = []
        node_list = []
        sub_node_list = []
        for v in sorted(g.nodes()):
            if v > len(self.data.g_list):
                rank_list.append(self.node_count[v - len(self.data.g_list)] / len(self.data.g_list))
                node_list.append(v)
        sorted_list = sorted(rank_list)
        a = int(len(sorted_list) * 0.9)
        threshold_num = sorted_list[a-1]
        for i in range(len(rank_list)):
            if rank_list[i] > threshold_num:
                sub_node_list.append(node_list[i])
        self.removed_nodes = sub_node_list
        count = 0
        label_mapping = {}
        for v in sorted(g.nodes()):
            if v in sub_node_list:
                count += 1
            else:
                label_mapping[v] = v - count
        for v in sub_node_list:
            g.remove_node(v)
        g = nx.relabel_nodes(g, label_mapping)
        return g

    @staticmethod
    def shift_right(l):
        if type(l) == int:
            return l
        elif type(l) == tuple:
            l = list(l)
            return tuple([l[-1]] + l[:-1])
        elif type(l) == list:
            return tuple([l[-1]] + l[:-1])
        else:
            print('ERROR!')

    @staticmethod
    def find_ring_weights(ring, g):
        weight_list = []
        for i in range(len(ring)-1):
            weight = g.get_edge_data(ring[i], ring[i+1])['weight']
            weight_list.append(weight)
        weight = g.get_edge_data(ring[-1], ring[0])['weight']
        weight_list.append(weight)
        return weight_list

