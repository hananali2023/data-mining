from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from graphframes import GraphFrame
import os
import sys
import time


def main():
    os.environ["PYSPARK_SUBMIT_ARGS"] =  "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    conf = SparkConf().setAppName("task1")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    spark.sparkContext.setLogLevel("ERROR")

    start_time = time.time()
    df = spark.read.csv(input_file_path, header=True, inferSchema=True)

    user_business_rdd = df.select("user_id", "business_id").rdd.map(lambda row: (row["user_id"], row["business_id"]))

    user_business_set_rdd = user_business_rdd.groupByKey().mapValues(set)

    user_pairs_rdd = user_business_set_rdd.cartesian(user_business_set_rdd)

    edges_rdd = (user_pairs_rdd.filter(lambda pair: pair[0][0] != pair[1][0]).filter(lambda pair: len(pair[0][1].intersection(pair[1][1])) >= filter_threshold).map(lambda pair: (pair[0][0], pair[1][0])).distinct())
    #edges = edges_rdd.collect()
    edges_df = edges_rdd.toDF(["src", "dst"])

    nodes_df = edges_df.select("src").union(edges_df.select("dst")).distinct().withColumnRenamed("src", "id")

    graph = GraphFrame(nodes_df, edges_df)


    communities = graph.labelPropagation(maxIter=5)

    sorted_communities = (communities.rdd.map(lambda row: (row["label"], row["id"])).groupByKey().mapValues(sorted).map(lambda x: x[1]).sortBy(lambda x: (len(x), x)).map(lambda x: [f"'{user}'" for user in x]).collect())

    with open(output_file_path, "w") as file:
        for community in sorted_communities:
            file.write(", ".join(community) + "\n")

    spark.stop()


if __name__ == "__main__":
    main()

