import sys
import time
from pyspark import SparkConf, SparkContext
from collections import defaultdict
from itertools import combinations

def find_betweenness(graph):
    edge_betweenness = defaultdict(float)
    for start_node in graph:
        parent = defaultdict(set)
        shortest = defaultdict(int)
        shortest[start_node] = 1
        levels = {start_node: 0}
        node_list = [start_node]
        while node_list:
            current = node_list.pop(0)
            for neighbor in graph[current]:
                if neighbor not in levels:
                    node_list.append(neighbor)
                    levels[neighbor] = levels[current] + 1
                if levels[neighbor] == levels[current] + 1:
                    parent[neighbor].add(current)
                    shortest[neighbor] += shortest[current]
        credit = defaultdict(lambda: 1.0)
        for node in sorted(parent.keys(), key=lambda n: levels[n], reverse=True):
            for par in parent[node]:
                node_fraction = (shortest[par] / shortest[node]) * credit[node]
                edge = tuple(sorted((par, node)))
                edge_betweenness[edge] += node_fraction
                credit[par] += node_fraction
    return sorted([(edge, round(score / 2, 5)) for edge, score in edge_betweenness.items()],key=lambda x: (-x[1], x[0]))

def new_communities(graph):
    visited, communities = set(), []
    for node in graph:
        if node not in visited:
            community, node_list = [], [node]
            while node_list:
                current = node_list.pop(0)
                if current not in visited:
                    visited.add(current)
                    community.append(current)
                    node_list.extend(graph[current] - visited)
            communities.append(sorted(community))
    return communities

def modularity(graph, coms, degree_map, m):
    mod = 0.0
    for com in coms:
        for i in range(len(com)):
            for j in range(i + 1, len(com)):
                a, b = com[i], com[j]
                if a not in degree_map or b not in degree_map:
                    continue
                Aab = 1 if b in graph[a] and a in graph[b] else 0
                mod += Aab - (degree_map[a] * degree_map[b]) / (2 * m)
    return mod / (2 * m)

def main():
    threshold = int(sys.argv[1])
    input_path = sys.argv[2]
    betweenness_out = sys.argv[3]
    community_out = sys.argv[4]

    conf = SparkConf().setAppName("task2")
    sc = SparkContext(conf=conf)
    start_time = time.time()

    raw_data = sc.textFile(input_path).map(lambda line: line.split(","))
    header = raw_data.first()
    data = raw_data.filter(lambda line: line != header)

    user_to_business = data.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set).collectAsMap()
    edges = [(user1, user2) for user1, user2 in combinations(user_to_business.keys(), 2)
             if len(user_to_business[user1].intersection(user_to_business[user2])) >= threshold]
    graph = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)

    edge_between = find_betweenness(graph)
    with open(betweenness_out, 'w') as f:
        for edge, score in edge_between:
            f.write(f"{edge}, {score}\n")

    node_degrees = {node: len(neighbors) for node, neighbors in graph.items()}

    total_edges = len(edges)
    best_modularity = -1
    best_communities = []

    while edge_between:
        max_betweenness = edge_between[0][1]
        remove_edges = [edge for edge, score in edge_between if score == max_betweenness]

        for a, b in remove_edges:
            graph[a].remove(b)
            graph[b].remove(a)


        communities = new_communities(graph)
        current_modularity = modularity(graph, communities, node_degrees, total_edges)

        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_communities = communities

        edge_between = find_betweenness(graph)

    with open(community_out, 'w') as f:
        for community in sorted(best_communities, key=lambda x: (len(x), x)):
            f.write(", ".join(f"'{node}'" for node in community) + "\n")

    sc.stop()

if __name__ == "__main__":
    main()


# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2.py 3 ../resource/asnlib/publicdata/ub_sample_data.csv betweenness.txt community.txt