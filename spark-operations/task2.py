import json
import sys
from pyspark import SparkConf, SparkContext
import time



def task2(input_file, output_file, n_partitions):

    # Spark Configuration
    conf = SparkConf().setAppName("task2")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")


    # Load reviews and business
    reviews = sc.textFile(input_file)
    business = reviews.map(lambda x: (json.loads(x)['business_id'], 1))

    # Paritioning
    partition_numbers = business.getNumPartitions()
    partition_sizes = business.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()

    # Repartitioning for customized results
    repartitioning = business.repartition(n_partitions)
    custom_partition_sizes = repartitioning.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()


    #top_10 = business.reduceByKey(lambda x, y: x + y).takeOrdered(10, key=lambda x: -x[1])

    # Default exe_time
    start_time = time.time()
    overall_time = time.time() - start_time

    # Custom exe_time
    custom_time = time.time()
    #top_10_custom = repartitioning.reduceByKey(lambda x, y: x + y).takeOrdered(10, key=lambda x: -x[1])
    custom_overall_time = time.time() - custom_time




    # All results
    results = {
        "default": {"n_partition": partition_numbers, "n_items": partition_sizes, "exe_time": overall_time},
        "customized": {"n_partition": int(n_partitions), "n_items": custom_partition_sizes, "exe_time": custom_overall_time}
    }


    # Write results into output file
    with open(output_file, 'w') as output_f:
        json.dump(results, output_f, indent=4)


    # Stop Spark!
    sc.stop()


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    n_partitions = int(sys.argv[3])
    task2(input_file, output_file, n_partitions)