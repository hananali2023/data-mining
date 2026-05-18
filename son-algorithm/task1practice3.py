import sys
import time
from pyspark import SparkConf, SparkContext
from collections import defaultdict, Counter
from itertools import chain, combinations

# NEED:
    # Hash function ?
    # PCY Algorithm
    # SON Algorithm


def pcy_pass(basket_iter, total_baskets, min_support, bucket_count):
    basket_list = list(basket_iter)

    support_threshold = len(basket_list) / total_baskets * min_support

    item_counter = Counter()
    bucket_counter = defaultdict(int)
    #bucket_count = defaultdict(int)

    # Bitmap ?
    bitmap = [0] * bucket_count

    for basket in basket_list:
        item_counter.update(basket)
        pairs = list(combinations(basket, 2))

        for pair in pairs:
            hashed_pair = (int(pair[0]) ^ int(pair[1])) % bucket_count
            bucket_counter[hashed_pair] += 1

    for bucket, count in bucket_counter.items():
        if count >= support_threshold:
            bitmap[bucket] = 1

    # Frequent Singles
    frequent_singles = [item for item, count in item_counter.items() if count >= support_threshold]
    frequent_candidates = [(tuple([item]), 1) for item in frequent_singles]

    #print(f"Singletons: {frequent_singles}")

    # Frequent Pairs
    frequent_pairs = [pair for pair in combinations(frequent_singles, 2)
                      if bitmap[(int(pair[0]) ^ int(pair[1])) % bucket_count] == 1]


    if not frequent_pairs:
        return []

    refined_baskets = [[item for item in basket if item in frequent_singles] for basket in basket_list]

    pair_support = defaultdict(int)
    for basket in refined_baskets:
        for pair in frequent_pairs:
            if set(pair).issubset(basket):
                pair_support[pair] += 1

    pair_support = {pair: count for pair, count in pair_support.items() if count >= support_threshold}
    frequent_candidates.extend([(pair, 1) for pair in pair_support.keys()])

    # Larger itemsets
    k = 3
    while pair_support:
        unique_singles = set(chain.from_iterable(pair_support.keys()))
        candidate_itemsets = list(combinations(unique_singles, k))

        itemset_support = defaultdict(int)
        for basket in refined_baskets:
            for itemset in candidate_itemsets:
                if set(itemset).issubset(basket):
                    itemset_support[itemset] += 1

        itemset_support = {
            itemset: count for itemset, count in itemset_support.items() if count >= support_threshold
        }

        frequent_candidates.extend([(itemset, 1) for itemset in itemset_support.keys()])

        pair_support = itemset_support
        k += 1

    #print(f"Frequnet candidates: {frequent_candidates}")

    return frequent_candidates


def counts_frequents(baskets, candidates):
    candidate_list = list(candidates)
    frequent_sets = []
    for basket in baskets:
        for itemset in candidate_list:
            if set(itemset).issubset(basket):
                frequent_sets.append(itemset)

    #print(f"Frequents: {frequent_sets}")

    return frequent_sets


def son_algorithm(baskets, support_threshold, bucket_count=1000):
    num_baskets = baskets.count()
    itemsets = baskets.values()

    # Phase 1: Candidates
    candidate_itemsets = itemsets.mapPartitions(
        lambda part: pcy_pass(part, num_baskets, support_threshold, bucket_count)
    ).reduceByKey(lambda x, y: x + y) \
     .map(lambda x: tuple(sorted(x[0]))) \
     .distinct().sortBy(lambda x: (len(x), x))

    candidates_grouped = candidate_itemsets.groupBy(lambda x: len(x)) \
                                           .mapValues(list) \
                                           .sortByKey() \
                                           .collectAsMap()

    candidates_list = candidate_itemsets.collect()

    # Phase 2: Count itemsets occurrences
    frequent_itemsets = itemsets.mapPartitions(
        lambda part: counts_frequents(part, candidates_list)).map(lambda itemset: (itemset, 1)).reduceByKey(lambda x, y: x + y) \
     .filter(lambda x: x[1] >= support_threshold).map(lambda x: tuple(sorted(x[0]))).sortBy(lambda x: (len(x), x)).groupBy(lambda x: len(x)) \
     .mapValues(list).sortByKey().collectAsMap()

    #print(f"Candidates: {candidates_grouped}")

    #print(f"Frequent: {frequent_itemsets}")

    return candidates_grouped, frequent_itemsets





def main(case_num, min_support, input_path, output_path):
    conf = SparkConf().setAppName("task1")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    start_time = time.time()
    raw_data = sc.textFile(input_path).filter(lambda row: row != "user_id,business_id")

    # Baskets based on case number!
    if case_num == 1:
        baskets = raw_data.map(lambda line: line.split(",")).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b).cache()
    elif case_num == 2:
        baskets = raw_data.map(lambda line: line.split(",")).map(lambda x: (x[1], [x[0]])).reduceByKey(lambda a, b: a + b).cache()


    # SON
    #print(f"SON Algorithm!")
    candidates, frequent_sets = son_algorithm(baskets, min_support)

    # Output to file
    with open(output_path, "w") as f:
        f.write("Candidates:\n")
        for item, itemsets in candidates.items():
            itemsets = ",".join(map(str, itemsets)).replace(",)", ")")
            f.write(f"{itemsets}\n\n")

        f.write("Frequent Itemsets:\n")
        for line, itemsets in frequent_sets.items():
            itemsets = ",".join(map(str, itemsets)).replace(",)", ")")
            f.write(f"{itemsets}")
            if line < len(frequent_sets):
                f.write("\n\n")

    print(f"Duration: {time.time() - start_time}")


    sc.stop()


if __name__ == "__main__":
    case_num = int(sys.argv[1])
    min_support = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]

    main(case_num, min_support, input_path, output_path)
