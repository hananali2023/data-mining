from pyspark import SparkContext, SparkConf
from sklearn.cluster import KMeans
import numpy as np
import math
from collections import defaultdict
import sys


def calculate_mahalanobis(point, cluster_summary):
    centroid = np.array(cluster_summary["SUM"]) / cluster_summary["N"]
    variance = np.array(cluster_summary["SUMSQ"]) / cluster_summary["N"] - (centroid ** 2)
    variance = np.where(variance > 0, variance, 1e-10)
    inv_cov = np.linalg.inv(np.diag(variance))
    delta = np.array(point) - centroid
    return np.sqrt(delta.T @ inv_cov @ delta)


def update_clusters(summary, point):
    summary["N"] += 1
    summary["SUM"] = np.add(summary["SUM"], point)
    summary["SUMSQ"] = np.add(summary["SUMSQ"], np.square(point))
    return summary


def summarize_clusters(cluster_assignments, points_map):
    vectors = {}
    for cluster_id, point_indices in cluster_assignments.items():
        points = [points_map[idx] for idx in point_indices]
        vectors[cluster_id] = {
            "N": len(points),
            "SUM": np.sum(points, axis=0),
            "SUMSQ": np.sum(np.square(points), axis=0),
            "POINTS": point_indices}
    return vectors


def k_means(data, num_clusters, ids=None):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
    cluster_assignments = defaultdict(list)
    if ids is None:
        ids = list(range(len(data)))
    for idx, cluster_label in enumerate(kmeans.labels_):
        cluster_assignments[cluster_label].append(ids[idx])
    return cluster_assignments


def assign_points(points, cluster_summaries, distance_threshold):
    assigned_points = defaultdict(list)
    unassigned_points = []
    for index, point in points.items():
        min_distance = float('inf')
        nearest_cluster = None
        for cluster_id, cluster_summary in cluster_summaries.items():
            distance = calculate_mahalanobis(point, cluster_summary)
            if distance < distance_threshold and distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster_id
        if nearest_cluster is not None:
            assigned_points[nearest_cluster].append(index)
            cluster_summaries[nearest_cluster] = update_clusters(cluster_summaries[nearest_cluster], point)
        else:
            unassigned_points.append(index)
    return assigned_points, unassigned_points


def merge_clusters(cs, ds, distance_threshold):
    used_cs_labels = set()
    for cs_label, cs_summary in list(cs.items()):
        if cs_label in used_cs_labels:
            continue
        closest_ds = None
        min_distance = float('inf')
        for ds_label, ds_summary in ds.items():
            distance = calculate_mahalanobis(cs_summary["SUM"] / cs_summary["N"], ds_summary)
            if distance < distance_threshold and distance < min_distance:
                min_distance = distance
                closest_ds = ds_label
        if closest_ds is not None:
            ds[closest_ds]["N"] += cs_summary["N"]
            ds[closest_ds]["SUM"] = np.add(ds[closest_ds]["SUM"], cs_summary["SUM"])
            ds[closest_ds]["SUMSQ"] = np.add(ds[closest_ds]["SUMSQ"], cs_summary["SUMSQ"])
            ds[closest_ds]["POINTS"].extend(cs_summary["POINTS"])
            del cs[cs_label]
            used_cs_labels.add(cs_label)
    return ds

def merge_cs_clusters(compression_set, distance_threshold):
    merged = True
    while merged:
        merged = False
        cs_keys = list(compression_set.keys())
        for i in range(len(cs_keys)):
            for j in range(i + 1, len(cs_keys)):
                c1, c2 = cs_keys[i], cs_keys[j]
                if c1 not in compression_set or c2 not in compression_set:
                    continue
                sum1 = compression_set[c1]
                sum2 = compression_set[c2]
                centroid1 = np.array(sum1["SUM"]) / sum1["N"]
                dist = calculate_mahalanobis(centroid1, sum2)
                if dist < distance_threshold:
                    sum2["N"] += sum1["N"]
                    sum2["SUM"] = np.add(sum2["SUM"], sum1["SUM"])
                    sum2["SUMSQ"] = np.add(sum2["SUMSQ"], sum1["SUMSQ"])
                    sum2["POINTS"].extend(sum1["POINTS"])
                    del compression_set[c1]
                    merged = True
                    break
                if merged:
                    break
    return compression_set

def process_data_chunk(chunk, discard_set, compression_set, retained_set, num_clusters, points_map):
    points = chunk
    sample_point = next(iter(points.values()))
    if isinstance(sample_point, int):
        sample_point = points_map[sample_point]
    distance_threshold = 2 * math.sqrt(len(sample_point))
    ds_assignments, remaining_points = assign_points(points, discard_set, distance_threshold)
    cs_assignments, new_retained_points = assign_points({point: points[point] for point in remaining_points}, compression_set, distance_threshold)
    retained_set.update(new_retained_points)
    if len(retained_set) >= 5 * num_clusters:
        rs_points = [points_map[point] for point in retained_set]
        rs_keys = list(retained_set)
        rs_clusters = k_means(rs_points, min(num_clusters * 5, len(rs_points)), rs_keys)
        unique_cluster_id = max(compression_set.keys(), default=-1) + 1 if compression_set else 0
        for cluster_id, members in rs_clusters.items():
            if len(members) > 1:
                compression_set[unique_cluster_id] = summarize_clusters({unique_cluster_id: members}, points_map)[unique_cluster_id]
                for member in members:
                    retained_set.discard(member)
                unique_cluster_id += 1
        compression_set = merge_cs_clusters(compression_set, distance_threshold)
    return discard_set, compression_set, retained_set


def bfr(input_file, num_clusters, output_file):
    global sc
    data_rdd = sc.textFile(input_file).map(lambda line: line.split(',')).map(lambda line: (int(line[0]), list(map(float, line[1:]))))
    points_map = data_rdd.collectAsMap()
    point_items = list(points_map.items())
    chunks = [dict() for _ in range(5)]
    for idx, (k, v) in enumerate(point_items):
        chunks[idx % 5][k] = v
    discard_set, compression_set, retained_set = {}, {}, set()
    intermediate_results = []
    clustering_results = {}
    for i in range(5):
        chunk = chunks[i]
        if i == 0:
            initial_k = min(2 * num_clusters, len(chunk))
            tmp_kmeans = KMeans(n_clusters=initial_k, random_state=0).fit(list(chunk.values()))
            singleton_ids = set()
            for label in set(tmp_kmeans.labels_):
                indices = [idx for idx, lbl in enumerate(tmp_kmeans.labels_) if lbl == label]
                if len(indices) == 1:
                    singleton_ids.add(list(chunk.keys())[indices[0]])
            RS = {idx: chunk[idx] for idx in singleton_ids}
            remaining = {k: v for k, v in chunk.items() if k not in RS}
            final_assignments = k_means(list(remaining.values()), num_clusters, list(remaining.keys()))
            discard_set = summarize_clusters(final_assignments, remaining)
            retained_set = set(RS.keys())
        else:
            discard_set, compression_set, retained_set = process_data_chunk(
                chunk, discard_set, compression_set, retained_set, num_clusters, points_map)
        intermediate_results.append((
            sum(cluster["N"] for cluster in discard_set.values()),
            len(compression_set),
            sum(cluster["N"] for cluster in compression_set.values()),
            len(retained_set)))
    distance_threshold = 2 * math.sqrt(len(point_items[0][1]))
    discard_set = merge_clusters(compression_set, discard_set, distance_threshold)
    for cluster_id, cluster in discard_set.items():
        for point_idx in cluster["POINTS"]:
            clustering_results[point_idx] = cluster_id
        for point_idx in retained_set:
            clustering_results[point_idx] = -1




    with open(output_file, 'w') as f:
        f.write("The intermediate results:\n")
        for round_idx, result in enumerate(intermediate_results, start=1):
            f.write(f"Round {round_idx}: {','.join(map(str, result))}\n")
        f.write("\nThe clustering results:\n")
        for point_idx, cluster_id in sorted(clustering_results.items()):
            f.write(f"{point_idx},{cluster_id}\n")


if __name__ == "__main__":
    conf = SparkConf().setAppName("task")
    sc = SparkContext(conf=conf)
    input_file = sys.argv[1]
    num_clusters = int(sys.argv[2])
    output_file = sys.argv[3]


    bfr(input_file, num_clusters, output_file)


