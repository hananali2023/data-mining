import sys
import time
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict



def generate_candidates(baskets, k):
    return baskets.flatMap(lambda basket: combinations(sorted(basket), k)).distinct()

# This function is outputting only the singles when all the candidates are being inputted. Something is wrong with the baskets
def count_itemsets(baskets, candidates):
    candidate_set = set(candidates)
    return baskets.flatMap(
        lambda basket: [tuple(sorted(itemset)) for size in range(1, len(max(candidates, key=len)) + 1)
                        for itemset in combinations(basket, size) if tuple(sorted(itemset)) in candidate_set]).map(
        lambda itemset: (itemset, 1)).reduceByKey(lambda a, b: a + b)


                                              #for itemset in combinations(basket, len(candidates[0])) if tuple(sorted(itemset)) in candidate_set])\


    #print(f"Baskets in count_itemsets: {baskets.collect()}")

    #return baskets


def has_frequent_subsets(candidate, frequent_itemsets, k):
    subsets = combinations(candidate, k - 1)
    for subset in subsets:
        if tuple(sorted(subset)) not in frequent_itemsets:
            #print(f"Pruned candidate {candidate} due to subset {subset}")
            return False
    return True


def a_priori(baskets_iter, global_support_threshold, total_basket_count):
    baskets = list(baskets_iter)
    #partition_support = (global_support_threshold / total_basket_count)

    new_support = len(baskets) / total_basket_count

    partition_support = new_support * global_support_threshold

    #print(f"Parition:{partition_support}")

    # Step 1: Frequent singles
    singletons = defaultdict(int)
    for basket in baskets:
        for item in set(basket):
            singletons[item] += 1

    frequent_singletons = {item for item, count in singletons.items() if count >= partition_support}

    #print(f"Frequent singles: {frequent_singletons}")


    frequent_itemsets = {tuple([item]): count for item, count in singletons.items() if count >= partition_support}
    candidates = list(frequent_itemsets.keys())

    #print(f"Candidates: {candidates}")

    all_candidates = list(candidates)
    k = 2

    # Step 2: Larger frequent itemsets
    while candidates:
        itemsets = set([tuple(sorted(set(c1).union(set(c2))))
                        for c1 in candidates for c2 in candidates
                        if len(set(c1).union(set(c2))) == k and c1[:-1] == c2[:-1] and c1[-1] < c2[-1]])

        itemsets = {itemset for itemset in itemsets if has_frequent_subsets(itemset, frequent_itemsets, k)}

        itemset_counts = defaultdict(int)
        for basket in baskets:
            basket_set = set(basket)
            for itemset in itemsets:
                if set(itemset).issubset(basket_set):
                    itemset_counts[itemset] += 1

        frequent_itemsets_k = {itemset: count for itemset, count in itemset_counts.items() if count >= partition_support}

        if not frequent_itemsets_k:
            break

        frequent_itemsets.update(frequent_itemsets_k)
        candidates = list(frequent_itemsets_k.keys())
        all_candidates.extend(candidates)

        k += 1

    return all_candidates




def son_algorithm(sc, case, support_threshold, input_file_path, output_file_path):
    start_time = time.time()

    data = sc.textFile(input_file_path)
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(","))

    # Case 1
    if case == 1:
        baskets = data.groupByKey().map(lambda x: list(set(x[1])))
    # Case 2
    elif case == 2:
        baskets = data.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: list(set(x[1])))
    else:
        print("Invalid case number")
        sys.exit(1)

    total_basket_count = baskets.count()

    # Phase 1: Candidates
    #number_of_partitions = baskets.getNumPartitions()


    #local_support_threshold = support_threshold / number_of_partitions
    partitioned_candidates = baskets.mapPartitions(lambda partition: a_priori(partition, support_threshold, total_basket_count))

    all_candidates = partitioned_candidates.distinct().collect()
    #print(f"All candidates: {all_candidates}")


    # Phase 2: Counts of all candidates
    global_frequent_itemsets = count_itemsets(baskets, all_candidates).filter(lambda x: x[1] >= support_threshold).map(lambda x: x[0]).collect()

    #print(f"Frequent: {global_frequent_itemsets}")


    #print(f"Global frequents: {global_frequent_itemsets}")

    duration = time.time() - start_time
    print(f"Duration: {duration}")

    return all_candidates, global_frequent_itemsets



def write_output_to_file(output_file, candidates, frequent_itemsets):
    # Custom sorting function to prioritize 100's first, then 90's, then other numbers in ascending order
    def sorting_itemsets(itemsets):
        def custom_sort_key(itemset):
            first_item = itemset[0]
            num = int(first_item)
            if 100 <= num < 200:
                return (1, num)
            elif 90 <= num < 100:
                return (2, num)
            else:
                return (3, num)

        return sorted(itemsets, key=lambda x: (len(x), custom_sort_key(x)))


    def format_itemsets(itemsets):
        result = ""
        current_length = 0
        current_group = []

        for itemset in itemsets:
            if len(itemset) != current_length:
                if current_group:
                    result += ", ".join(current_group) + "\n\n"
                current_group = []
                current_length = len(itemset)

            if len(itemset) == 1:
                current_group.append(f"('{itemset[0]}')")
            else:
                current_group.append(f"{itemset}")

        if current_group:
            result += ", ".join(current_group) + "\n\n"

        return result

    with open(output_file, 'w') as f:
        f.write("Candidates:\n")

        candidates_sorted = sorting_itemsets(candidates)
        f.write(format_itemsets(candidates_sorted))

        f.write("Frequent Itemsets:\n")

        frequent_itemsets_sorted = sorting_itemsets(frequent_itemsets)
        f.write(format_itemsets(frequent_itemsets_sorted))








if __name__ == "__main__":

    case_number = int(sys.argv[1])
    support_threshold = float(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    sc = SparkContext(appName="task1")

    sc.setLogLevel('WARN')

    all_candidates, global_frequent_itemsets = son_algorithm(sc, case_number, support_threshold, input_file_path, output_file_path)

    write_output_to_file(output_file_path, all_candidates, global_frequent_itemsets)
    sc.stop()
