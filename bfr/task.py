from pyspark import SparkContext, SparkConf
from sklearn.cluster import KMeans
import sys
import numpy as np
from math import sqrt


def validate_data(data_rdd):
    def is_valid_row(row):
        try:
            numeric_row = [float(x) for x in row]
            return True, numeric_row
        except ValueError:
            return False, None

    cleaned = data_rdd.map(lambda row: is_valid_row(row)).filter(lambda x: x[0]).map(lambda x: x[1])
    return cleaned


def mahalanobis(point, cluster_stats):
    mean = cluster_stats["SUM"] / cluster_stats["N"]
    covariance = np.diag(cluster_stats["SUMSQ"] / cluster_stats["N"] - mean**2)
    inv_covariance = np.linalg.inv(covariance)
    return sqrt(np.dot(np.dot((point - mean), inv_covariance), (point - mean)))


def new_stats(cluster_points_rdd):
    cluster_stats = {}
    cluster_stats["N"] = cluster_points_rdd.count()
    cluster_stats["SUM"] = cluster_points_rdd.reduce(lambda x, y: np.add(x, y))
    cluster_stats["SUMSQ"] = cluster_points_rdd.map(lambda x: np.square(x)).reduce(lambda x, y: np.add(x, y))
    return cluster_stats


def new_points(data_rdd, clusters, threshold, feature_len):
    def closest_cluster(point):
        min_distance = float('inf')
        closest_cluster = None
        for cluster_id, stats in clusters.items():
            distance = mahalanobis(point, stats)
            if distance < threshold and distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        return closest_cluster, point

    assigned = data_rdd.map(closest_cluster)
    unassigned = assigned.filter(lambda x: x[0] is None).map(lambda x: x[1])
    assigned = assigned.filter(lambda x: x[0] is not None)
    return assigned, unassigned


def merged_clusters(clusters, threshold, feature_len):
    merged = {}
    keys = list(clusters.keys())
    for i, key1 in enumerate(keys):
        if key1 not in clusters:
            continue
        for key2 in keys[i + 1:]:
            if key2 in clusters and mahalanobis(
                    clusters[key1]["SUM"] / clusters[key1]["N"],
                    clusters[key2]) < threshold:
                clusters[key1]["N"] += clusters[key2]["N"]
                clusters[key1]["SUM"] += clusters[key2]["SUM"]
                clusters[key1]["SUMSQ"] += clusters[key2]["SUMSQ"]
                del clusters[key2]
        merged[key1] = clusters[key1]
    return merged


def calculate_kmeans(data_rdd, k, feature_len):
    data = np.array(data_rdd.collect(), dtype=float)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = kmeans.labels_
    clusters = {}
    for label in set(labels):
        cluster_points = data[labels == label]
        clusters[label] = new_stats(SparkContext.getOrCreate().parallelize(cluster_points))
    return clusters


def main(input_file, n_clusters, output_file):
    conf = SparkConf().setAppName("BFR Algorithm").set("spark.executor.memory", "4g").set("spark.executor.cores", "2")
    sc = SparkContext(conf=conf)

    data_rdd = sc.textFile(input_file).map(lambda line: line.split(','))
    data_rdd = validate_data(data_rdd)

    shuffled_data_rdd = data_rdd.zipWithIndex().map(lambda x: (np.random.random(), x[0])).sortByKey().map(lambda x: x[1])
    total_records = shuffled_data_rdd.count()
    chunk_size = total_records // 5

    DS = {}
    CS = {}
    RS = []
    results = []
    feature_len = len(data_rdd.take(1)[0])

    for round_num in range(1, 6):
        start_index = (round_num - 1) * chunk_size
        end_index = round_num * chunk_size if round_num < 5 else total_records

        chunk_rdd = shuffled_data_rdd.zipWithIndex().filter(lambda x: start_index <= x[1] < end_index).map(lambda x: x[0])

        if round_num == 1:
            # initialize DS, CS, RS
            initial_clusters = calculate_kmeans(chunk_rdd, n_clusters * 5, feature_len)
            rs_rdd = chunk_rdd.filter(lambda point: len(initial_clusters) == 1)
            RS = rs_rdd.collect()

            ds_rdd = chunk_rdd.filter(lambda point: point not in RS)
            DS_clusters = calculate_kmeans(ds_rdd, n_clusters, feature_len)
            DS.update(DS_clusters)
        else:
            assigned_to_DS_rdd, remaining_rdd = new_points(chunk_rdd, DS, threshold=2 * sqrt(feature_len), feature_len=feature_len)
            for cluster_id, point in assigned_to_DS_rdd.collect():
                DS[cluster_id]["N"] += 1
                DS[cluster_id]["SUM"] += point
                DS[cluster_id]["SUMSQ"] += point**2

            assigned_to_CS_rdd, RS_rdd = new_points(remaining_rdd, CS, threshold=2 * sqrt(feature_len), feature_len=feature_len)
            for cluster_id, point in assigned_to_CS_rdd.collect():
                CS[cluster_id]["N"] += 1
                CS[cluster_id]["SUM"] += point
                CS[cluster_id]["SUMSQ"] += point**2

            rs_points = RS_rdd.collect()
            new_clusters = calculate_kmeans(sc.parallelize(rs_points), n_clusters * 5, feature_len)
            RS = [point for cluster_id, points in new_clusters.items() if len(points) == 1]
            CS.update({cid: new_stats(sc.parallelize(points)) for cid, points in new_clusters.items() if len(points) > 1})
            CS = new_clusters(CS, threshold=2 * sqrt(feature_len), feature_len=feature_len)
        results.append(f"Round {round_num}: {sum([v['N'] for v in DS.values()])}, {len(CS)}, "f"{sum([v['N'] for v in CS.values()])}, {len(RS)}")
    DS = new_clusters({**DS, **CS}, threshold=2 * sqrt(feature_len), feature_len=feature_len)

    with open(output_file, 'w') as f:
        f.write("The intermediate results:\n")
        f.write("\n".join(results))
        f.write("\n\nThe clustering results:\n")
        for cluster_id, stats in DS.items():
            f.write(f"Cluster {cluster_id}: {stats}\n")


if __name__ == "__main__":
    input_file = sys.argv[1]
    n_clusters = int(sys.argv[2])
    output_file = sys.argv[3]
    main(input_file, n_clusters, output_file)
