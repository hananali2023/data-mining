from pyspark import SparkContext, SparkConf
from sklearn.cluster import KMeans
import numpy as np
import math
from collections import defaultdict
import sys
import time
from sklearn.preprocessing import StandardScaler

# Function to parse data
def parse_data(line):
    try:
        parts = line.split(",")
        return int(parts[0]), list(map(float, parts[1:]))
    except Exception as e:
        print(f"Error in parse_data: {e}")
        raise

# Mahalanobis distance function
def calculate_mahalanobis(point, cluster_summary):
    try:
        mean = np.array(cluster_summary["SUM"]) / cluster_summary["N"]
        variance = np.array(cluster_summary["SUMSQ"]) / cluster_summary["N"] - mean ** 2
        epsilon = 1e-10  # zero or very small variance
        variance = np.where(variance > 0, variance, epsilon)
        inv_cov = np.linalg.inv(np.diag(variance))
        delta = np.array(point) - mean
        return np.sqrt(delta.T @ inv_cov @ delta)
    except Exception as e:
        print(f"Error in calculate_mahalanobis: {e}")
        raise

# Update cluster stats when a point is added
def update_clusters(cluster_summary, point):
    try:
        cluster_summary["N"] += 1
        cluster_summary["SUM"] = np.add(cluster_summary["SUM"], point)
        cluster_summary["SUMSQ"] = np.add(cluster_summary["SUMSQ"], np.square(point))
        return cluster_summary
    except Exception as e:
        print(f"Error in update_clusters: {e}")
        raise

# Summarize clusters from K-Means results
def summarize_clusters(cluster_assignments, points_map):
    try:
        cluster_stats = {}
        for cluster_id, point_indices in cluster_assignments.items():
            points = [points_map[idx] for idx in point_indices]
            cluster_stats[cluster_id] = {
                "N": len(points),
                "SUM": np.sum(points, axis=0),
                "SUMSQ": np.sum(np.square(points), axis=0),
                "POINTS": point_indices,
            }
        return cluster_stats
    except Exception as e:
        print(f"Error in summarize_clusters: {e}")
        raise

# Run K-Means clustering
def k_means(data, num_clusters):
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, init="k-means++").fit(data_scaled)
        cluster_assignments = defaultdict(list)
        for idx, cluster_label in enumerate(kmeans.labels_):
            cluster_assignments[cluster_label].append(idx)
        return cluster_assignments
    except Exception as e:
        print(f"Error in k_means: {e}")
        raise

# Assign points to clusters using Mahalanobis distance
def assign_points(points, cluster_summaries, distance_threshold):
    try:
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
    except Exception as e:
        print(f"Error in assign_points: {e}")
        raise

# Merge compression set (CS) clusters into discard set (DS)
def merge_clusters(cs, ds, distance_threshold):
    try:
        for cs_label, cs_summary in list(cs.items()):
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

        return ds
    except Exception as e:
        print(f"Error in merge_clusters: {e}")
        raise

# Process a chunk of data
def process_data_chunk(chunk, discard_set, compression_set, retained_set, num_clusters):
    try:
        points = {index: point for index, point in enumerate(chunk)}
        distance_threshold = 2 * math.sqrt(len(points[0]))

        # Assign points to discard set (DS)
        ds_assignments, remaining_points = assign_points(points, discard_set, distance_threshold)

        # Assign remaining points to compression set (CS)
        cs_assignments, new_retained_points = assign_points(
            {point: points[point] for point in remaining_points}, compression_set, distance_threshold
        )

        # Update retained set with points that could not be assigned
        retained_set.update(new_retained_points)

        # Run K-Means on retained set if it is not empty
        if retained_set:
            rs_points = [points[point] for point in retained_set]
            rs_clusters = k_means(rs_points, min(num_clusters * 5, len(rs_points)))

            for cluster_id, members in rs_clusters.items():
                if len(members) > 1:
                    compression_set[cluster_id] = summarize_clusters({cluster_id: members}, points)[cluster_id]
                    for member in members:
                        retained_set.discard(member)

        return discard_set, compression_set, retained_set
    except Exception as e:
        print(f"Error in process_data_chunk: {e}")
        raise

# BFR function
def bfr(input_file, num_clusters, output_file):
    try:
        global sc

        # Load input data
        data_rdd = sc.textFile(input_file).map(lambda line: line.split(',')).map(lambda line: (int(line[0]), list(map(float, line[1:]))))
        points_map = data_rdd.collectAsMap()
        data_chunks = list(points_map.values())
        chunk_size = len(data_chunks) // 5

        # Initialize sets and results
        discard_set, compression_set, retained_set = {}, {}, set()
        intermediate_results = []
        clustering_results = {}

        # Process each chunk of data
        for i in range(5):
            try:
                chunk = data_chunks[i * chunk_size: (i + 1) * chunk_size] if i < 4 else data_chunks[i * chunk_size:]

                if i == 0:
                    initial_clusters = k_means(chunk, num_clusters * 5)
                    clusters = {cid: idx for cid, idx in initial_clusters.items() if len(idx) > 1}
                    retained_set = {idx for idxs in initial_clusters.values() if len(idxs) == 1 for idx in idxs}
                    discard_set = summarize_clusters(clusters, points_map)
                else:
                    discard_set, compression_set, retained_set = process_data_chunk(
                        chunk, discard_set, compression_set, retained_set, num_clusters
                    )

                # Record intermediate results
                intermediate_results.append((
                    sum(cluster["N"] for cluster in discard_set.values()),
                    len(compression_set),
                    sum(cluster["N"] for cluster in compression_set.values()),
                    len(retained_set)
                ))
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                raise

        # Final merge of CS into DS
        distance_threshold = 2 * math.sqrt(len(data_chunks[0]))
        discard_set = merge_clusters(compression_set, discard_set, distance_threshold)

        # Assign clustering results
        for cluster_id, cluster in discard_set.items():
            for point_idx in cluster["POINTS"]:
                clustering_results[point_idx] = cluster_id

        for point_idx in retained_set:
            clustering_results[point_idx] = -1

        # Write results to output file
        with open(output_file, 'w') as f:
            f.write("The intermediate results:\n")
            for round_idx, result in enumerate(intermediate_results, start=1):
                f.write(f"Round {round_idx}: {','.join(map(str, result))}\n")
            f.write("\nThe clustering results:\n")
            for point_idx, cluster_id in sorted(clustering_results.items()):
                f.write(f"{point_idx},{cluster_id}\n")
    except Exception as e:
        print(f"Error in BFR function: {e}")
        raise

if __name__ == "__main__":
    try:
        conf = SparkConf().setAppName("task")
        sc = SparkContext(conf=conf)
        sc.setLogLevel('WARN')

        input_file = sys.argv[1]
        num_clusters = int(sys.argv[2])
        output_file = sys.argv[3]

        bfr(input_file, num_clusters, output_file)
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        sc.stop()
