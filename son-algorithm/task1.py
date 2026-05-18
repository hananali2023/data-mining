import sys
import time
from collections import Counter
from itertools import chain, combinations
from pyspark import SparkConf, SparkContext


def case_baskets(data, case):
    if case == 1:
        baskets = data.map(lambda line: line.strip().split(",")).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b)
    elif case == 2:
        baskets = data.map(lambda line: line.strip().split(",")).map(lambda x: (x[1], [x[0]])).reduceByKey(lambda a, b: a + b)
    return baskets


def hash_function(item1, item2, num_buckets):
    hash_val = int(item1) ^ int(item2)
    return hash_val % num_buckets


def pcy_algorithm(baskets, total_num_baskets, support, num_buckets):
    candidates = []
    baskets = list(baskets)

    support_threshold= len(baskets) / total_num_baskets
    partition_support_threshold = support_threshold * support

    item_counts = Counter()
    hash_counts = Counter()

    bitmap = [0] * num_buckets

    for basket in baskets:
        item_counts.update(basket)

        pairs = list(combinations(basket, 2))

        hashes = [hash_function(item[0], item[1], num_buckets) for item in pairs]
        hash_counts.update(hashes)

    for h, count in hash_counts.items():
        if count >= partition_support_threshold:
            bitmap[h] = 1

    frequent_singles = []
    for item, count in item_counts.items():
        if count >= partition_support_threshold:
            frequent_singles.append(item)
            candidates.append((tuple([item]), 1))

    frequent_pairs = []
    for item in combinations(frequent_singles, 2):
        hash_val = hash_function(item[0], item[1], num_buckets)
        if bitmap[hash_val] == 1:
            frequent_pairs.append(item)

    if len(frequent_pairs) == 0:
        return []

    baskets = [[item for item in basket if item in frequent_singles] for basket in baskets]

    candidate_dict = {pair: 0 for pair in frequent_pairs}

    for basket in baskets:
        for pair in frequent_pairs:
            if set(pair).issubset(basket):
                candidate_dict[pair] += 1

    candidate_dict = {pair: count for pair, count in candidate_dict.items() if count >= partition_support_threshold}

    candidates += [(pair, 1) for pair in candidate_dict.keys()]

    val_candidates = candidate_dict.keys()

    k = 3
    while len(val_candidates) > 0:
        unq_singles = set(chain.from_iterable(val_candidates))
        itemsets = list(combinations(unq_singles, k))

        candidate_dict = {itemset: 0 for itemset in itemsets}

        for basket in baskets:
            for itemset in itemsets:
                if set(itemset).issubset(basket):
                    candidate_dict[itemset] += 1

        candidate_dict = {itemset: count for itemset, count in candidate_dict.items() if count >= partition_support_threshold}

        candidates += [(itemset, 1) for itemset in candidate_dict.keys()]

        val_candidates = candidate_dict.keys()

        k += 1

    return candidates

def frequents(baskets, candidates):
    candidates = list(candidates)
    frequent_itemsets = []

    for basket in baskets:
        for itemset in candidates:
            if set(itemset).issubset(basket):
                frequent_itemsets.append(itemset)

    return frequent_itemsets


def son_algorithm(baskets, support, num_buckets=1000):
    num_baskets = baskets.count()

    items = baskets.values()

    candidates = (items.mapPartitions(lambda chunk: pcy_algorithm(chunk, num_baskets, support, num_buckets))
        .reduceByKey(lambda x, y: x + y)
        .map(lambda x: tuple(sorted(x[0])))
        .distinct()
        .sortBy(lambda x: (len(x), x)))

    candidates_map = (candidates.groupBy(lambda x: len(x)).mapValues(list).sortByKey().collectAsMap())
    candidates = candidates.collect()

    frequent_itemsets = (items.mapPartitions(lambda chunk: frequents(chunk, candidates)).map(lambda itemset: (itemset, 1)).reduceByKey(lambda x, y: x + y)
        .filter(lambda x: x[1] >= support).map(lambda x: tuple(sorted(x[0]))).sortBy(lambda x: (len(x), x))
        .groupBy(lambda x: len(x)).mapValues(list).sortByKey().collectAsMap())

    return candidates_map, frequent_itemsets


def main(case_number, support, input_file_path, output_file_path):
    conf = SparkConf().setAppName("Task 1")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()
        data = spark.textFile(input_file_path)
        data = data.filter(lambda row: row != "user_id,business_id")

        baskets = case_baskets(data, case_number).cache()
        candidates, frequent_itemsets = son_algorithm(baskets, support, num_buckets=1000)

        with open(output_file_path, "w") as f:
            f.write("Candidates:\n")
            for _, itemsets in candidates.items():
                itemsets = ",".join(map(str, itemsets)).replace(",)", ")")
                f.write(f"{itemsets}\n\n")
            f.write("Frequent Itemsets:\n")
            for line, itemsets in frequent_itemsets.items():
                itemsets = ",".join(map(str, itemsets)).replace(",)", ")")
                f.write(f"{itemsets}")
                if line < len(frequent_itemsets):
                    f.write("\n\n")

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        spark.stop()


if __name__ == "__main__":
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    main(case_number, support, input_file_path, output_file_path)
