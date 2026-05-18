from pyspark import SparkConf, SparkContext
import os, shutil, sys, time, math
from collections import defaultdict
from itertools import combinations


def preprocess_data(sc, input_file, output_file):
    temp_output_dir = "temp_preprocessed_data"
    if os.path.exists(temp_output_dir):
        try:
            shutil.rmtree(temp_output_dir)
        except Exception as e:
            print(f"Error removing existing directory: {e}")
    raw_data = sc.textFile(input_file).filter(
        lambda
            row: row.strip() and row != '"TRANSACTION_DT","CUSTOMER_ID","AGE_GROUP","PIN_CODE","PRODUCT_SUBCLASS","PRODUCT_ID","AMOUNT","ASSET","SALES_PRICE"'
    )

    def process_row(x):
        fields = x.split(',')
        try:
            date_parts = fields[0].replace('"', '').split('/')
            date_customer_id = date_parts[0] + "/" + date_parts[1] + "/" + date_parts[2][2:] + "-" + str(
                int(fields[1].replace('"', '')))
            product_id = str(int(fields[5].replace('"', '')))
            return date_customer_id, product_id
        except (IndexError, ValueError) as e:
            return None

    preprocessed_data = raw_data.map(process_row).filter(lambda x: x is not None)
    header = "DATE-CUSTOMER_ID,PRODUCT_ID"
    output_rdd = sc.parallelize([header]).union(preprocessed_data.map(lambda x: f"{x[0]},{x[1]}"))
    output_rdd.coalesce(1).saveAsTextFile(temp_output_dir)
    part_file = None
    for file in os.listdir(temp_output_dir):
        if file.startswith("part-"):
            part_file = os.path.join(temp_output_dir, file)
            break
    if part_file and os.path.exists(part_file):
        shutil.move(part_file, output_file)
    shutil.rmtree(temp_output_dir)


def get_frequent_itemsets(candidates, baskets, threshold):
    frequents = []
    for candidate in candidates:
        count = 0
        for basket in baskets:
            if candidate.issubset(basket):
                count += 1
        if count >= threshold:
            frequents.append(candidate)
    return frequents


# Updated apriori_partition using k-2 as join condition for k>2 and caching sorted candidates.
def apriori_partition(baskets_iter, init_singles, part_threshold):
    baskets = list(baskets_iter)
    all_candidates = []
    current = init_singles
    k = 2
    while current:
        freq = get_frequent_itemsets(current, baskets, part_threshold)
        if not freq:
            break
        all_candidates.extend(freq)
        new_candidates = []
        # Precompute sorted tuples for candidates to avoid repeated sorting.
        sorted_candidates = [tuple(sorted(list(c))) for c in freq]
        for i in range(len(freq)):
            for j in range(i + 1, len(freq)):
                # For k==2, join all pairs; for k>2, join only if first (k-2) items match.
                if k == 2 or sorted_candidates[i][:k - 2] == sorted_candidates[j][:k - 2]:
                    joined = freq[i].union(freq[j])
                    if joined not in new_candidates:
                        new_candidates.append(joined)
        current = new_candidates
        k += 1
    return all_candidates


def count_partition(baskets_iter, candidate_list):
    baskets = list(baskets_iter)
    counts = defaultdict(int)
    for candidate in candidate_list:
        cand_set = set(candidate)
        for basket in baskets:
            if cand_set.issubset(basket):
                counts[candidate] += 1
    return list(counts.items())


def son_algorithm(sc, baskets_rdd, global_support, min_basket_size):
    filtered = baskets_rdd.filter(lambda x: len(x[1]) > min_basket_size).cache()
    total_baskets = filtered.count()
    # Convert baskets to sets and repartition to 200 for better parallelism.
    basket_sets = filtered.values().map(lambda lst: set(lst)).repartition(200).cache()
    num_parts = basket_sets.getNumPartitions()
    part_sup = math.ceil(global_support / float(num_parts))

    singles = basket_sets.flatMap(lambda b: list(b)).distinct().map(lambda item: frozenset([item])).collect()

    part_candidates = basket_sets.mapPartitions(
        lambda part: apriori_partition(list(part), singles, part_sup)
    ).map(lambda cand: (tuple(sorted(cand)), 1)) \
        .reduceByKey(lambda a, b: a) \
        .map(lambda pair: pair[0]).collect()

    candidates_b = sc.broadcast(part_candidates)

    global_counts = basket_sets.mapPartitions(
        lambda part: count_partition(list(part), candidates_b.value)
    ).reduceByKey(lambda a, b: a + b) \
        .filter(lambda pair: pair[1] >= global_support) \
        .map(lambda pair: pair[0]).collect()

    def group_itemsets(itemsets):
        groups = defaultdict(set)
        for it in itemsets:
            groups[len(it)].add(tuple(sorted(it)))
        return {k: sorted(list(v)) for k, v in groups.items()}

    candidate_group = group_itemsets(part_candidates)
    frequent_group = group_itemsets(global_counts)
    return candidate_group, frequent_group


def write_results(out_file, candidate_groups, freq_groups):
    with open(out_file, 'w') as f:
        f.write("Candidates:\n")
        for size in sorted(candidate_groups.keys()):
            f.write(", ".join(str(item) for item in candidate_groups[size]).replace(",)", ")") + "\n\n")
        f.write("Frequent Itemsets:\n")
        for size in sorted(freq_groups.keys()):
            f.write(", ".join(str(item) for item in freq_groups[size]).replace(",)", ")") + "\n\n")


def main():
    conf = SparkConf().setAppName("task2")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    try:
        start = time.time()
        filter_thresh = int(sys.argv[1])
        global_sup = float(sys.argv[2])
        input_csv = sys.argv[3]
        result_file = sys.argv[4]
        preprocessed_file = "preprocessed_data.csv"

        preprocess_data(sc, input_csv, preprocessed_file)

        raw = sc.textFile(preprocessed_file).filter(lambda row: row.strip() and row != "DATE-CUSTOMER_ID,PRODUCT_ID")
        baskets = raw.map(lambda line: line.split(',')) \
            .filter(lambda parts: len(parts) >= 2) \
            .map(lambda parts: (parts[0].strip(), [parts[1].strip()])) \
            .reduceByKey(lambda a, b: a + b)

        candidate_groups, frequent_groups = son_algorithm(sc, baskets, global_sup, filter_thresh)

        write_results(result_file, candidate_groups, frequent_groups)
        print("Duration:", time.time() - start)
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
