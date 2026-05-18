from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors
import numpy as np
import sys

sc = SparkContext("local", "task")
output_file_path = sys.argv[3]

with open(output_file_path, 'w') as f:
    f.write("The intermediate results:\n")

input_file = sys.argv[1]
full_rdd = sc.textFile(input_file)


def parse_line(line):
    parts = line.split()
    return Vectors.dense([float(x) for x in parts[2:]])



def mahalanobis_distance(point, cluster_summary):
    point = np.array(point)

    N = cluster_summary["N"]
    SUM = np.array(cluster_summary["SUM"])
    SUMSQ = np.array(cluster_summary["SUMSQ"])

    centroid = SUM / N
    variance = (SUMSQ / N) - (centroid ** 2)
    variance[variance == 0] = 1e-10

    diff = point - centroid
    mahalanobis_dist = np.sqrt(np.sum((diff ** 2) / variance))

    return mahalanobis_dist



def mahalanobis_distance_between_clusters(cluster1, cluster2):
    N1, SUM1, SUMSQ1 = cluster1["N"], np.array(cluster1["SUM"]), np.array(cluster1["SUMSQ"])
    N2, SUM2, SUMSQ2 = cluster2["N"], np.array(cluster2["SUM"]), np.array(cluster2["SUMSQ"])

    centroid1 = SUM1 / N1
    centroid2 = SUM2 / N2

    variance1 = (SUMSQ1 / N1) - (centroid1 ** 2)
    variance2 = (SUMSQ2 / N2) - (centroid2 ** 2)

    average_variance = (variance1 + variance2) / 2
    average_variance[average_variance == 0] = 1e-10

    diff = centroid1 - centroid2
    mahalanobis_dist = np.sqrt(np.sum((diff ** 2) / average_variance))

    return mahalanobis_dist


def assign_to_cluster(point, cluster_summaries, threshold=2):
    min_distance = float("inf")
    assigned_cluster = -1

    for cluster_id, summary in cluster_summaries:
        distance = mahalanobis_distance(point, summary)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            assigned_cluster = cluster_id

    return (assigned_cluster, point)



def merge_cs_clusters(cs_summary, threshold=2):
    merged_clusters = []
    visited = set()

    for i, (cluster_id1, cluster1) in enumerate(cs_summary):
        if cluster_id1 in visited:
            continue

        merged_cluster = cluster1.copy()
        merged_cluster_id = cluster_id1

        for j, (cluster_id2, cluster2) in enumerate(cs_summary):
            if i == j or cluster_id2 in visited:
                continue

            distance = mahalanobis_distance_between_clusters(cluster1, cluster2)
            if distance < threshold:
                merged_cluster["N"] += cluster2["N"]
                merged_cluster["SUM"] = np.add(merged_cluster["SUM"], cluster2["SUM"])
                merged_cluster["SUMSQ"] = np.add(merged_cluster["SUMSQ"], cluster2["SUMSQ"])
                visited.add(cluster_id2)

        merged_clusters.append((merged_cluster_id, merged_cluster))
        visited.add(cluster_id1)

    return merged_clusters

initial_sample_rdd = full_rdd.sample(withReplacement=False, fraction=0.2, seed=42)
initial_data_rdd = initial_sample_rdd.map(parse_line)

n_clusters = int(sys.argv[2])
kmeans_model_initial = KMeans.train(initial_data_rdd, k=n_clusters, maxIterations=10, initializationMode="k-means||", seed=42)
initial_predictions = initial_data_rdd.map(lambda point: (kmeans_model_initial.predict(point), point))


def calculate_sums(cluster_points):
    cluster_id, points = cluster_points
    N = len(points)
    SUM = np.sum(points, axis=0)
    SUMSQ = np.sum(np.square(points), axis=0)
    return (cluster_id, {"N": N, "SUM": SUM, "SUMSQ": SUMSQ})

grouped_points = initial_predictions.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list)
ds_summary_results = grouped_points.map(calculate_sums).collect()


remaining_data = full_rdd.subtract(initial_sample_rdd)


cs_summary_results = []

chunk_fraction = 0.1
round_num = 1

while not remaining_data.isEmpty() and round_num <= 3:
    print(f"Starting round {round_num}")

    sampled_rdd = remaining_data.sample(withReplacement=False, fraction=chunk_fraction,
                                        seed=np.random.randint(1000)).cache()
    new_data_rdd = sampled_rdd.map(parse_line)
    remaining_data = remaining_data.subtract(sampled_rdd)

    ds_summary_broadcast = sc.broadcast(ds_summary_results)
    assigned_to_ds_rdd = new_data_rdd.map(lambda point: assign_to_cluster(point, ds_summary_broadcast.value))
    ds_points = assigned_to_ds_rdd.filter(lambda x: x[0] != -1).map(lambda x: (x[0], x[1]))
    remaining_rs_points = assigned_to_ds_rdd.filter(lambda x: x[0] == -1).map(lambda x: x[1])

    ds_points_grouped = ds_points.groupByKey().mapValues(list)
    for cluster_id, points in ds_points_grouped.collect():
        for idx, (cid, summary) in enumerate(ds_summary_results):
            if cid == cluster_id:
                summary["N"] += len(points)
                summary["SUM"] = np.add(summary["SUM"], np.sum(np.array(points), axis=0))
                summary["SUMSQ"] = np.add(summary["SUMSQ"], np.sum(np.square(np.array(points)), axis=0))
                ds_summary_results[idx] = (cid, summary)
                break

    cs_summary_broadcast = sc.broadcast(cs_summary_results)
    assigned_to_cs_rdd = remaining_rs_points.map(lambda point: assign_to_cluster(point, cs_summary_broadcast.value))
    cs_points = assigned_to_cs_rdd.filter(lambda x: x[0] != -1).map(lambda x: (x[0], x[1]))
    remaining_rs_points_after_cs = assigned_to_cs_rdd.filter(lambda x: x[0] == -1).map(lambda x: x[1])

    cs_points_grouped = cs_points.groupByKey().mapValues(list)
    for cluster_id, points in cs_points_grouped.collect():
        if cluster_id not in [c[0] for c in cs_summary_results]:
            cs_summary_results.append((cluster_id, {"N": len(points), "SUM": np.sum(np.array(points), axis=0),
                                                    "SUMSQ": np.sum(np.square(np.array(points)), axis=0)}))
        else:
            for i, (cid, summary) in enumerate(cs_summary_results):
                if cid == cluster_id:
                    summary["N"] += len(points)
                    summary["SUM"] = np.add(summary["SUM"], np.sum(np.array(points), axis=0))
                    summary["SUMSQ"] = np.add(summary["SUMSQ"], np.sum(np.square(np.array(points)), axis=0))
                    cs_summary_results[i] = (cid, summary)

    rs_points_final = remaining_rs_points_after_cs

    num_points_ds = sum([summary["N"] for _, summary in ds_summary_results])
    num_clusters_cs = len(cs_summary_results)
    num_points_rs = rs_points_final.count()

    with open(output_file_path, 'a') as f:
        f.write(f"Round {round_num}: {num_points_ds},{num_clusters_cs},{num_points_rs}\n")

    rs_points_final = rs_points_final.map(lambda x: Vectors.dense(x) if not isinstance(x, Vectors) else x).cache()
    rs_points_final = rs_points_final.filter(lambda x: not any(np.isnan(x)) and not any(np.isinf(x)))

    if not rs_points_final.isEmpty():
        num_points_rs = rs_points_final.count()
        large_k = min(3, 5 * len(cs_summary_results))

        k = min(num_points_rs, large_k)
        if k > 0:
            try:
                kmeans_model_large_k_rs = KMeans.train(rs_points_final, k=k, maxIterations=10, seed=42)
                print(f"Round {round_num}: K-means clustering completed successfully on RS.")

                rs_predictions = rs_points_final.map(lambda point: (kmeans_model_large_k_rs.predict(point), point))
                rs_cluster_counts = rs_predictions.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
                rs_cluster_counts_dict = rs_cluster_counts.collectAsMap()

                new_rs_points = rs_predictions.filter(lambda x: rs_cluster_counts_dict[x[0]] == 1).map(lambda x: x[1])
                new_cs_points = rs_predictions.filter(lambda x: rs_cluster_counts_dict[x[0]] > 1).map(lambda x: x[1])

                for cluster_id, point in new_cs_points.collect():
                    if cluster_id not in [c[0] for c in cs_summary_results]:
                        cs_summary_results.append(
                            (cluster_id, {"N": 1, "SUM": np.array(point), "SUMSQ": np.square(np.array(point))}))
                    else:
                        for i, (cid, summary) in enumerate(cs_summary_results):
                            if cid == cluster_id:
                                summary["N"] += 1
                                summary["SUM"] = np.add(summary["SUM"], np.array(point))
                                summary["SUMSQ"] = np.add(summary["SUMSQ"], np.square(np.array(point)))
                                cs_summary_results[i] = (cid, summary)

            except Exception as e:
                print(f"Failed to train K-means model on RS: {e}")
    else:
        print(f"Round {round_num}: RS is empty, skipping K-means clustering.")

    round_num += 1

total_points = full_rdd.count()
assigned_ds = sum(summary["N"] for (_, summary) in ds_summary_results)
assigned_cs = sum(summary["N"] for (_, summary) in cs_summary_results)
assigned_total = assigned_ds + assigned_cs
print(f"Assigned: {assigned_total}/{total_points}")

sc.stop()

with open(output_file_path, 'a') as f:
    f.write("\nThe clustering results:\n")
    if ds_summary_results:
        for cluster_id, summary in ds_summary_results:
            f.write(f"{cluster_id},0\n")
    if cs_summary_results:
        for cluster_id, summary in cs_summary_results:
            f.write(f"{cluster_id},1\n")


