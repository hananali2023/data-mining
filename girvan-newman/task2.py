import sys
import time
from pyspark import SparkConf, SparkContext
from collections import defaultdict
from itertools import combinations


def betweenness(graph):
    edge_btw = defaultdict(float)
    for node in graph:
        parent_node = defaultdict(set)
        shortest = defaultdict(int)
        shortest[node] = 1
        levels = {node: 0}
        node_list = [node]
        while node_list:
            node = node_list.pop(0)
            for neighbor in graph[node]:
                if neighbor not in levels:
                    node_list.append(neighbor)
                    levels[neighbor] = levels[node] + 1
                if levels[neighbor] == levels[node] + 1:
                    parent_node[neighbor].add(node)
                    shortest[neighbor] += shortest[node]
        node_credit = {n: 1.0 for n in graph}
        for n in sorted(parent_node.keys(), key=lambda x: len(parent_node[x]), reverse=True):
            for p in parent_node[n]:
                credit = (shortest[p] / shortest[n]) * node_credit[n]
                edge = tuple(sorted((p, n)))
                edge_btw[edge] += credit
                node_credit[p] += credit
    return sorted([(a, round(b / 2, 5)) for a, b in edge_btw.items()], key=lambda x: (-x[1], x[0]))


def communities(graph):
    visited = set()
    coms = []
    def dfs(node, community):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                community.append(current)
                stack.extend(graph[current] - visited)
    for node in graph:
        if node not in visited:
            com = []
            dfs(node, com)
            coms.append(sorted(com))
    return coms


def modularity(graph, coms, degree_map, m):
    mod = 0.0
    for com in coms:
        for i in range(len(com)):
            for j in range(i + 1, len(com)):
                a, b = com[i], com[j]
                Aab = 1 if b in graph[a] and a in graph[b] else 0
                mod += Aab - (degree_map[a] * degree_map[b]) / (2 * m)
    return mod / (2 * m)


def main():
    filter_threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    betweenness_output = sys.argv[3]
    community_output = sys.argv[4]

    conf = SparkConf().setAppName("task2")
    sc = SparkContext(conf=conf)

    start_time = time.time()

    raw_data = sc.textFile(input_file).map(lambda line: line.split(","))
    header = raw_data.first()
    data = raw_data.filter(lambda row: row != header)


    user_business_rdd = data.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set)
    edges = [(user1, user2) for user1, user2 in combinations(user_business_rdd.keys(), 2)
             if len(user_business_rdd[user1].intersection(user_business_rdd[user2])) >= filter_threshold]

    edge_list = defaultdict(set)
    for a, b in edges:
        edge_list[a].add(b)
        edge_list[b].add(a)
    graph, original_edges = edge_list, edges

    between = betweenness(graph)
    with open(betweenness_output, 'w') as f:
        for edge, value in between:
            f.write(f"('{edge[0]}', '{edge[1]}'), {value}\n")

    original_degree = {node: len(neighbors) for node, neighbors in graph.items()}
    m = len(original_edges)
    max_modularity = -1
    optimal_community = []

    while True:
        between = betweenness(graph)
        if not between:
            break
        max_betweenness = between[0][1]
        high_betweenness_edges = [edge for edge, btw in between if btw == max_betweenness]

        for a, b in high_betweenness_edges:
            graph[a].remove(b)
            graph[b].remove(a)

        community = communities(graph)

        modular = modularity(graph, community, original_degree, m)

        if modular > max_modularity:
            max_modularity = modular
            optimal_community = community

        if len(between) == 0:
            break


    with open(community_output, 'w') as f:
        for com in sorted(optimal_community, key=lambda x: (len(x), x)):
            f.write(", ".join(f"'{node}'" for node in com) + "\n")

    print("Execution time:", time.time() - start_time)

    sc.stop()


if __name__ == "__main__":
    main()

