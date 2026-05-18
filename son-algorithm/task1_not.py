import sys
import time
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict





def compute_num_buckets(total_basket_count):
    # You can adjust the factor here to control the number of buckets
    return max(1000, total_basket_count // 10)  # Example heuristic: use 1/10th of basket count or 1000 (whichever is larger)


def pcy_pass_1(baskets_iter, global_support_threshold, total_basket_count):
    baskets = list(baskets_iter)
    num_buckets = compute_num_buckets(total_basket_count)

    new_support = len(baskets) / total_basket_count

    partition_support = new_support * global_support_threshold
    #partition_support = global_support_threshold / total_basket_count


    # Step 1: Count single items (frequent singletons)
    singletons = defaultdict(int)
    for basket in baskets:
        for item in set(basket):
            singletons[item] += 1

    # Step 2: Filter frequent singletons (L1)
    frequent_singletons = {item for item, count in singletons.items() if count >= partition_support}

    # Step 3: Hash pairs into buckets
    bucket_counts = defaultdict(int)
    for basket in baskets:
        basket_items = [item for item in basket if item in frequent_singletons]
        for pair in combinations(basket_items, 2):
            hash_value = hash(pair) % num_buckets
            bucket_counts[hash_value] += 1

    # Debugging: Print structure
    print(f"Returning from pcy_pass_1: {list(frequent_singletons)}, {dict(bucket_counts)}")

    # Return frequent singletons and bucket counts
    return list(frequent_singletons), dict(bucket_counts)


# Pass 2 of PCY: Count pairs in frequent buckets
def pcy_pass_2(baskets_iter, frequent_singletons, bucket_counts, global_support_threshold, total_basket_count):
    baskets = list(baskets_iter)
    partition_support = global_support_threshold / total_basket_count

    # Dynamically calculate number of buckets
    num_buckets = compute_num_buckets(total_basket_count)

    # Step 1: Create a bit vector for frequent buckets
    bit_vector = [1 if bucket_counts[hash_value] >= partition_support else 0 for hash_value in bucket_counts]

    # Step 2: Generate candidate pairs from frequent singletons and check the hash
    candidate_pairs = defaultdict(int)
    for basket in baskets:
        basket_items = [item for item in basket if item in frequent_singletons]
        for pair in combinations(basket_items, 2):
            hash_value = hash(pair) % num_buckets
            if bit_vector[hash_value] == 1:  # Only count pairs in frequent buckets
                candidate_pairs[pair] += 1

    # Filter candidate pairs that meet the support threshold
    frequent_pairs = {pair: count for pair, count in candidate_pairs.items() if count >= partition_support}

    return frequent_pairs


def son_pcy_phase_1(baskets, support_threshold, total_basket_count):
    # Map partitions and apply pcy_pass_1 to each partition
    partitioned_candidates_and_buckets = baskets.mapPartitions(
        lambda partition: pcy_pass_1(partition, support_threshold, total_basket_count)
    )

    # Now `partitioned_candidates_and_buckets` should be an RDD of tuples (frequent_singletons, bucket_counts)

    # Collect frequent singletons from all partitions
    all_frequent_singletons = partitioned_candidates_and_buckets.flatMap(lambda x: x[0]).distinct().collect()

    # Collect and merge bucket counts from all partitions
    all_bucket_counts = partitioned_candidates_and_buckets.map(lambda x: x[1]).reduce(
        lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
    )

    return all_frequent_singletons, all_bucket_counts


# Phase 2: Count the global occurrence of the candidate itemsets using PCY
def son_pcy_phase_2(baskets, frequent_singletons, bucket_counts, support_threshold, total_basket_count):
    # Map partitions and apply pcy_pass_2 to each partition
    partitioned_frequent_pairs = baskets.mapPartitions(
        lambda partition: pcy_pass_2(partition, frequent_singletons, bucket_counts, support_threshold,
                                     total_basket_count)
    )

    # Collect and aggregate counts from all partitions
    all_frequent_pairs = partitioned_frequent_pairs.reduceByKey(lambda a, b: a + b).filter(
        lambda x: x[1] >= support_threshold).collect()

    return all_frequent_pairs

def son_algorithm(sc, case, support_threshold, input_file_path, output_file_path):
    start_time = time.time()

    data = sc.textFile(input_file_path)
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(","))

    # Case 1: group by user ID
    if case == 1:
        baskets = data.groupByKey().map(lambda x: list(set(x[1])))
    # Case 2: group by business ID
    elif case == 2:
        baskets = data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: list(set(x[1])))
    else:
        print("Invalid case number")
        sys.exit(1)

    total_basket_count = baskets.count()

    # Phase 1: Use PCY to generate candidate itemsets (frequent singletons and bucket counts)
    frequent_singletons, bucket_counts = son_pcy_phase_1(baskets, support_threshold, total_basket_count)
    print(f"Frequent singletons from phase 1: {frequent_singletons}")
    print(f"Bucket counts from phase 1: {bucket_counts}")

    # Phase 2: Use PCY to count the candidate itemsets across all baskets
    global_frequent_itemsets = son_pcy_phase_2(baskets, frequent_singletons, bucket_counts, support_threshold,
                                               total_basket_count)
    print(f"Global frequent itemsets: {global_frequent_itemsets}")

    duration = time.time() - start_time
    print(f"Duration: {duration}")

    return frequent_singletons, global_frequent_itemsets


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: spark-submit task1_not.py <case_number> <support> <input_file_path> <output_file_path>")
        sys.exit(-1)

    case_number = int(sys.argv[1])
    support_threshold = float(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    sc = SparkContext(appName="SONAlgorithmWithPCY")

    frequent_singletons, global_frequent_itemsets = son_algorithm(sc, case_number, support_threshold, input_file_path,
                                                                  output_file_path)

    # Writing output to file (can be implemented based on your needs)
    # For example: write_output_to_file(output_file_path, frequent_singletons, global_frequent_itemsets)

    sc.stop()


# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task1_not.py 1 4 ../resource/asnlib/publicdata/small2.csv task1.json