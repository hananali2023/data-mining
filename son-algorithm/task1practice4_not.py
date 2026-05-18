import sys
import time
from collections import defaultdict
from itertools import combinations, chain
from pyspark import SparkConf, SparkContext

def create_buckets(data, case_num):
    if case_num == 1:
        case_baskets = data.map(lambda line: line.strip().split(",")).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b)
    elif case_num == 2:
        case_baskets = data.map(lambda line: line.strip().split(",")).map(lambda x: (x[1], [x[0]])).reduceByKey(lambda a, b: a + b)

    return case_baskets

def hash_function(num1, num2, bucket_count):
    return (hash(str(num1)) ^ hash(str(num2))) % bucket_count

def pcy(basket_rdd, support, bucket_count):
    # Step 1: Count frequency of individual items (singletons) and hash pairs into buckets
    item_frequency_rdd = basket_rdd.flatMap(lambda basket: basket).map(lambda item: (item, 1)).reduceByKey(lambda x, y: x + y)

    # Step 2: Filter singletons by support threshold
    frequent_singletons_rdd = item_frequency_rdd.filter(lambda x: x[1] >= support).map(lambda x: x[0])

    # Step 3: Hash item pairs into buckets and count
    hash_bucket_rdd = basket_rdd.flatMap(lambda basket: [(hash_function(num1, num2, bucket_count), 1) for num1, num2 in combinations(basket, 2)])
    hash_bucket_count_rdd = hash_bucket_rdd.reduceByKey(lambda x, y: x + y)

    # Step 4: Generate bitmap for frequent pairs based on support threshold
    bitmap_rdd = hash_bucket_count_rdd.map(lambda x: (x[0], 1 if x[1] >= support else 0)).collectAsMap()

    # Step 5: Generate candidate pairs using frequent singletons and bitmap
    candidate_pairs_rdd = frequent_singletons_rdd.cartesian(frequent_singletons_rdd) \
        .filter(lambda x: x[0] < x[1] and bitmap_rdd.get(hash_function(x[0], x[1], bucket_count), 0) == 1)

    # Step 6: Count the occurrences of candidate pairs in baskets
    pair_count_rdd = basket_rdd.flatMap(lambda basket: [(tuple(sorted(pair)), 1) for pair in combinations(sorted(basket), 2)]) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] >= support) \
        .map(lambda x: x[0])

    # Step 7: Combine singletons and valid pairs as initial frequent itemsets
    all_candidates_rdd = frequent_singletons_rdd.map(lambda x: (x,)) \
        .union(pair_count_rdd)

    # Step 8: Generate larger itemsets (triples, quadruples, etc.)
    k = 3
    frequent_sets_rdd = pair_count_rdd  # Start with frequent pairs

    while not frequent_sets_rdd.isEmpty():
        # Generate k-itemsets by combining k-1 itemsets
        candidate_k_itemsets_rdd = frequent_sets_rdd.flatMap(
            lambda itemset: [tuple(sorted(set(itemset).union({x}))) for x in frequent_singletons_rdd.collect() if
                             x not in itemset and len(set(itemset).union({x})) == k])

        # Count the occurrences of k-itemsets in baskets
        frequent_sets_rdd = basket_rdd.flatMap(
            lambda basket: [(itemset, 1) for itemset in candidate_k_itemsets_rdd.collect() if
                            set(itemset).issubset(basket)]) \
            .reduceByKey(lambda x, y: x + y) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: x[0])

        if not frequent_sets_rdd.isEmpty():
            all_candidates_rdd = all_candidates_rdd.union(frequent_sets_rdd)  # Union with previous itemsets
        k += 1

    # Convert all lists to tuples before collecting
    print(f"Candidates PCY: {all_candidates_rdd.map(lambda x: tuple(x)).collect()}")
    return all_candidates_rdd
  # Return all candidates as RDD


def find_frequent_items(basket_rdd, candidate_rdd, support):
    frequent_itemsets_rdd = basket_rdd.flatMap(
        lambda basket: [(itemset, 1) for itemset in candidate_rdd.collect() if set(itemset).issubset(basket)]) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] >= support) \
        .map(lambda x: x[0])

    print(f"Frequents function: {frequent_itemsets_rdd.take(5)}")
    return frequent_itemsets_rdd


def son(baskets, support, bucket_count=10000):
    # First pass: generate candidates using PCY
    candidates = pcy(baskets, support, bucket_count)

    # Second pass: find frequent itemsets from the candidates
    frequent_itemsets = find_frequent_items(baskets, candidates, support)

    return candidates.collect(), frequent_itemsets.collect()




# def find_frequent_items(basket_chunk, candidate_sets, support):
#     frequent_itemsets = defaultdict(int)
#
#     # Count the occurrences of candidate itemsets in baskets
#     candidate_counts = defaultdict(int)
#     for basket in basket_chunk:
#         for itemset in candidate_sets:
#             if set(itemset).issubset(basket):
#                 candidate_counts[tuple(sorted(itemset))] += 1
#
#     # Only keep the itemsets that meet the support threshold
#     frequent_itemsets = [itemset for itemset, count in candidate_counts.items() if count >= support]
#
#     print(f"Find frequent: {frequent_itemsets}")
#     print(f"Type find frequent: {type(frequent_itemsets)}")
#
#     return frequent_itemsets



def task1(case, support, input_file, output_file):
    conf = SparkConf().setAppName("task1")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    start_time = time.time()
    data = sc.textFile(input_file).filter(lambda row: row != "user_id,business_id")
    baskets = create_buckets(data, case).cache()
    print(f"Sample baskets: {baskets.take(5)}")

    candidates, frequent_itemsets = son(baskets, support)

    # Write to output file
    with open(output_file, "w") as f:
        f.write("Candidates:\n")
        # Group candidates by size (length of itemset)
        candidates_by_size = defaultdict(list)
        for itemset in candidates:
            candidates_by_size[len(itemset)].append(itemset)

        for size in sorted(candidates_by_size.keys()):
            for itemset in sorted(candidates_by_size[size]):
                f.write(f"{itemset}\n")
            f.write("\n")

        f.write("Frequent Itemsets:\n")
        for size in sorted(frequent_itemsets.keys()):
            for itemset in sorted(frequent_itemsets[size]):
                f.write(f"{itemset}\n")
            f.write("\n")


    duration = time.time() - start_time
    print(f"Duration: {duration:.2f}")

    sc.stop()


if __name__ == "__main__":
    case = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    task1(case, support, input_file, output_file)