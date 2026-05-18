import sys
import time
import os
import shutil
import math
from pyspark import SparkConf, SparkContext
from collections import defaultdict, Counter
from itertools import combinations




def hash_items(item1, item2, bucket_count):
    clean_item1 = int(item1.strip().replace('"', ''))
    clean_item2 = int(item2.strip().replace('"', ''))
    return (clean_item1 ^ clean_item2) % bucket_count





def preprocess_tafeng_data(sc, input_file, output_file):
    temp_output_dir = "temp_preprocessed_data"

    if os.path.exists(temp_output_dir):
        try:
            shutil.rmtree(temp_output_dir)
        except Exception as e:
            print(f"Error removing existing directory: {e}")

    raw_data = sc.textFile(input_file).filter(lambda row: row.strip() and row != '"TRANSACTION_DT","CUSTOMER_ID","AGE_GROUP","PIN_CODE","PRODUCT_SUBCLASS","PRODUCT_ID","AMOUNT","ASSET","SALES_PRICE"')

    def process_row(x):
        fields = x.split(',')
        try:
            date_parts = fields[0].replace('"', '').split('/')
            # if len(date_parts) < 3:
            #     return None
            date_customer_id = date_parts[0] + "/" + date_parts[1] + "/" + date_parts[2][2:] + "-" + str(int(fields[1].replace('"', '')))
            product_id = str(int(fields[5].replace('"', '')))
            return date_customer_id, product_id
        except (IndexError, ValueError) as e:
            print(f"Error processing row: {fields}, Error: {e}")
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



def frequent_candidates(candidates, baskets, counts):
    frequent = []
    for cand in candidates:
        count = 0
        for basket in baskets:
            if cand.issubset(basket):
                count += 1
        if count >= counts:
            frequent.append(cand)
    return frequent

def apriori(baskets_iters, in_candidates, counts):
    baskets = list(baskets_iters)
    all_freq=[]
    current_candidates = in_candidates
    k = 2
    while current_candidates:
        frequent_cands = frequent_candidates(current_candidates, baskets, counts)
        if not frequent_cands:
            break
        all_freq.append(frequent_cands)
        next_candidates = []
        for i in range(len(frequent_cands)):
            for j in range(i+1, len(frequent_cands)):
                list_1 = sorted(list(frequent_cands[i]))
                list_2 = sorted(list(frequent_cands[j]))
                if list_1[:k-2] == list_2[:k-2]:
                    new_candidate = frequent_cands[i].union(frequent_cands[j])
                    if new_candidate not in current_candidates:
                        next_candidates.append(new_candidate)
        current_candidates = next_candidates
        k += 1
    return all_freq


def count_candidates(basket_iters, candidates):
    baskets = list(basket_iters)
    frequent_sets = []
    for cand in candidates:
        count = 0
        for basket in baskets:
            if cand.issubset(basket):
                count += 1
        if count > 0:
            frequent_sets.append(cand)
    return frequent_sets





# def pcy_pass(basket_iter, total_baskets, min_support, bucket_count):
#     basket_list = list(basket_iter)
#     support_threshold = (len(basket_list) / total_baskets) * min_support
#
#     item_counter = Counter()
#     bucket_counter = Counter()
#     bitmap = [0] * bucket_count
#
#     for basket in basket_list:
#         item_counter.update(basket)
#         for pair in combinations(basket, 2):
#             try:
#                 b = hash_items(pair[0], pair[1], bucket_count)
#             except Exception:
#                 continue
#             bucket_counter[b] += 1
#
#     for bucket, count in bucket_counter.items():
#         if count >= support_threshold:
#             bitmap[bucket] = 1
#
#     # Singles frequents
#     frequent_singles = [item for item, count in item_counter.items() if count >= support_threshold]
#     frequent_candidates = [((item), 1) for item in frequent_singles]
#
#
#     pair_support = defaultdict(int)
#     for basket in basket_list:
#         filtered = [item for item in basket if item in frequent_singles]
#         for pair in combinations(filtered, 2):
#             if bitmap[hash_items(pair[0], pair[1], bucket_count)] == 1:
#                 pair_support[tuple(sorted(pair))] += 1
#
#     for pair, count in pair_support.items():
#         if count >= support_threshold:
#             frequent_candidates.append((pair, 1))
#
#     return frequent_candidates


def son_algorithm(sc, baskets, support_threshold, filtered_threshold, bucket_count=1000):
    filtered_baskets = baskets.filter(lambda x: len(x[1]) > filtered_threshold).cache()
    num_baskets = filtered_baskets.count()
    # itemsets = filtered_baskets.values()
    basket_sets = filtered_baskets.values().map(lambda x: set(x))
    num_parts = basket_sets.getNumPartitions()
    part_support = math.ceil(support_threshold / float(num_parts))

    singles = basket_sets.flatMap(lambda basket: list(basket)).distinct().map(lambda x: frozenset([x])).collect()

    part_candidates = basket_sets.mapPartitions(lambda x: apriori(list(x), singles, part_support)).flatMap(lambda x: [x]).map(lambda x: (tuple(sorted(x)), 1)).reduceByKey(lambda a, b: a).map(lambda x: x[0]).collect()

    candidates_b = sc.broadcast(part_candidates)

    global_frequents = basket_sets.mapPartitions(lambda part: count_candidates(list(part), candidates_b.value)).map(lambda x: (tuple(sorted(x)), 1)).reduceByKey(lambda a, b: a + b).filter(lambda x: x[1] >= support_threshold).map(lambda x: x[0]).collect()

    candidate_group = {}
    for cand in part_candidates:
        size = len(cand)
        candidate_group.setdefault(size, set()).add(tuple(sorted(cand)))
    freq_group = {}
    for cand in global_frequents:
        size = len(cand)
        freq_group.setdefault(size, set()).add(tuple(sorted(cand)))


    for size in candidate_group:
        candidate_group[size] = sorted(list(candidate_group[size]))
    for size in freq_group:
        freq_group[size] = sorted(list(freq_group[size]))

    return candidate_group, freq_group


    # candidates = itemsets.mapPartitions(lambda part: pcy_pass(part, num_baskets, support_threshold, bucket_count)).reduceByKey(lambda x, y: x + y)\
    #  .map(lambda x: tuple(sorted(x[0]))).distinct().sortBy(lambda x: (len(x), x))
    #
    #
    # candidates_group = candidates.groupBy(lambda x: len(x)).mapValues(lambda a: sorted(list(a))).sortByKey().collect()
    # candidates_grouped = {size: groups for size, groups in candidates_group}
    # candidates_list = candidates.collect()
    # candidates_list_b = sc.broadcast(candidates_list)
    #
    # # Phase 2: Frequent itemsets w/ threshold
    # frequent_itemsets = itemsets.mapPartitions(
    #     lambda part: count_frequent_itemsets(list(part), candidates_list_b.value)
    # ).map(lambda itemset: (itemset, 1)).reduceByKey(lambda x, y: x + y) \
    #  .filter(lambda x: x[1] >= support_threshold).map(lambda x: tuple(sorted(x[0]))) \
    #  .groupBy(lambda x: len(x)).mapValues(lambda a: sorted(list(a))).collect()
    # frequent_itemset_grouped = {size: groups for size, groups in frequent_itemsets}
    #
    #
    # return candidates_grouped, frequent_itemset_grouped


# # Counter function
# def count_frequent_itemsets(baskets, candidates):
#     #candidate_list = list(candidates)
#     frequent_sets = []
#     for basket in baskets:
#         for itemset in candidates:
#             if set(itemset).issubset(basket):
#                 frequent_sets.append(itemset)
#     return frequent_sets


def write_output_to_file(output_file, candidates, frequent_itemsets):
    with open(output_file, 'w') as f:
        f.write("Candidates:\n")

        for item_len in sorted(candidates.keys()):
            if item_len == 1:
                itemsets_grouped = [f"('{itemset[0]}')" for itemset in candidates[item_len]]
            else:
                itemsets_grouped = [str(itemset).replace('"', '') for itemset in candidates[item_len]]
            f.write(", ".join(itemsets_grouped) + "\n\n")

        f.write("Frequent Itemsets:\n")

        for item_len in sorted(frequent_itemsets.keys()):
            if item_len == 1:
                itemsets_grouped = [f"('{itemset[0]}')" for itemset in frequent_itemsets[item_len]]
            else:
                itemsets_grouped = [str(itemset).replace('"', '') for itemset in frequent_itemsets[item_len]]
            f.write(", ".join(itemsets_grouped) + "\n\n")



"""
Homework Assignment 2
Task 2

Read data from the Ta-Feng dataset, pre-process it and implement the SON algorithm
with Apriori algorithm to find frequent item-sets.
"""

import os
import sys

from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce
from itertools import chain, combinations, groupby
from operator import add
from pyspark import SparkConf, SparkContext


def parse_args():
    if len(sys.argv) < 4:
        # expected arguments: script path, dataset path, output file path
        print('ERR: Expected three arguments: (case number, support, input file path, output file path).')
        exit(1)

    # read program arguments
    params['app_name'] = 'hw2-task2'
    params['threshold'] = int(sys.argv[1])
    params['support'] = int(sys.argv[2])
    params['in_file'] = sys.argv[3]
    params['out_file'] = sys.argv[4]
    params['processed_out_file'] = './processed.csv'
    return params


def write_csv(header_list, rows):
    with open(params['processed_out_file'], 'w') as file_handle:
        file_handle.write(','.join(header_list))
        file_handle.write('\n')
        for row in rows:
            file_handle.write(','.join(row))
            file_handle.write('\n')


def combine_ids_from_row(data_row, header_dict):
    row_transaction_date = data_row[header_dict["TRANSACTION_DT"]]
    row_transaction_date = row_transaction_date[: -4] + row_transaction_date[-2:]
    row_customer_id = data_row[header_dict["CUSTOMER_ID"]]
    row_product_id = str(int(data_row[header_dict["PRODUCT_ID"]]))

    combined_id = '{}-{}'.format(row_transaction_date, row_customer_id)
    return combined_id, row_product_id


def get_header_dict(file_name, encoding='utf-8'):
    header_dict = defaultdict(int)
    file_header_row = []
    with open(file_name, 'r', encoding=encoding) as file_handle:
        file_header_row = file_handle.readline().replace('"', '').strip().split(',')
        for idx, header in enumerate(file_header_row):
            header_dict[header] = idx
    return header_dict, file_header_row


def parse_raw_data():
    header_dict, file_header_row = get_header_dict(params['in_file'], 'utf-8-sig')
    raw_data_rdd = sc.textFile(params['in_file']) \
        .map(lambda line: line.replace('"', '').strip().split(',')) \
        .filter(lambda line: line != file_header_row) \
        .map(lambda line: combine_ids_from_row(line, header_dict))
    write_csv(['DATE-CUSTOMER_ID', 'PRODUCT_ID'], raw_data_rdd.collect(), )


def parse_processed_data():
    header_dict, file_header_row = get_header_dict(params['processed_out_file'], 'utf-8')
    file_header_tuple = tuple(file_header_row)
    processed_data_rdd = sc.textFile(params['processed_out_file']) \
        .map(lambda line: tuple(line.split(','))) \
        .filter(lambda line: line != file_header_tuple)
    return processed_data_rdd


def filter_processed_data():
    processed_data_rdd = processed_rdd.groupByKey()
    filtered_data_rdd = processed_data_rdd \
        .filter(lambda id_list: len(id_list[1]) > params['threshold']) \
        .map(lambda id_list: (id_list[0], set(id_list[1])))
    return filtered_data_rdd


def write_item_sets_by_count(item_sets_by_size, header, mode='w'):
    with open(params['out_file'], mode) as file_handle:
        file_handle.write(header)
        file_handle.write('\n')
        for size, candidates in item_sets_by_size.items():
            if len(candidates) == 0:
                file_handle.write('\n\n')
            if size == 1:
                for candidate in candidates[: -1]:
                    file_handle.write('(\'{}\'),'.format(candidate))
                file_handle.write('(\'{}\')\n\n'.format(candidates[-1]))
            else:
                for candidate in candidates[: -1]:
                    file_handle.write('{},'.format(candidate))
                file_handle.write(str(candidates[-1]))
                file_handle.write('\n\n')


def get_frequents_in_chunk(
        transactions,
        chunk_support,
        chunk_frequent_prev,
        frequent_item_set_size
):
    frequents = list()
    candidates = combinations(chunk_frequent_prev, 2)
    if frequent_item_set_size == 2:
        candidates = map(lambda pair: tuple(sorted(pair)), combinations(chunk_frequent_prev, 2))
    if frequent_item_set_size > 2:
        candidates = filter(
            lambda candidate_set: len(candidate_set) == frequent_item_set_size,
            map(lambda candidate_pair: tuple(sorted(set(candidate_pair[0]).union(candidate_pair[1]))), candidates)
        )
    for candidate in set(candidates):
        candidate_count = 0
        for transaction in transactions:
            if set(candidate).issubset(transaction):
                candidate_count += 1
            if candidate_count >= chunk_support:
                frequents.append(candidate)
                break

    return frequents


def apriori(chunk):
    # create a list of item sets from transaction chunk/partition
    transactions = list(map(lambda transaction: transaction[1], chunk))
    # calculate the adjusted support for this chunk/partition
    chunk_support = params['support'] * len(transactions) / total_transaction_count
    # initialize by calculating frequent singletons in the chunk/partition
    chunk_frequent_current = list(map(
        lambda item_count: item_count[0],
        filter(
            lambda item_count: item_count[1] >= chunk_support,
            Counter(chain.from_iterable(map(
                lambda transaction_item_set: list(transaction_item_set),
                transactions
            ))).items()
        )))
    frequent_item_set_size = 1

    # store the frequent item sets in a dictionary keyed by size
    chunk_frequents_comprehensive = defaultdict(list)
    chunk_frequents_comprehensive[frequent_item_set_size] = chunk_frequent_current

    # iteratively find frequent item sets for bugger sizes from the chunks
    while True:
        # increment frequent item set size
        frequent_item_set_size += 1
        # get chunks of the bigger size
        chunk_frequent_current = get_frequents_in_chunk(
            transactions,
            chunk_support,
            chunk_frequent_current,
            frequent_item_set_size
        )
        # quit when no more frequent item sets of the current size are found
        if len(chunk_frequent_current) == 0:
            break
        # add frequent item sets to the dictionary
        for item in chunk_frequent_current:
            chunk_frequents_comprehensive[frequent_item_set_size].append(item)

    # return the found frequent item sets of all sizes
    return list(map(lambda frequents: (frequents[0], frequents[1]), chunk_frequents_comprehensive.items()))


def get_frequents(chunk, candidates_by_size):
    candidates_by_count = defaultdict(int)
    candidate_counts = []
    # create a list of item sets from transaction chunk/partition
    transactions = list(map(lambda transaction: transaction[1], chunk))
    # calculate the adjusted support for this chunk/partition
    chunk_support = params['support'] * len(transactions) / total_transaction_count

    for size, candidate_item_sets in candidates_by_size.items():
        for candidate_item_set in candidate_item_sets:
            for transaction in transactions:
                item_set = set(candidate_item_set) if type(candidate_item_set) == tuple else {candidate_item_set}
                if set(item_set).issubset(transaction):
                    candidates_by_count[candidate_item_set] += 1
    for candidate_item_set, count in candidates_by_count.items():
        candidate_counts.append((candidate_item_set, count))

    return candidate_counts


def execute_son():
    # find candidates from individual chunks - first phase
    candidates_rdd = filtered_rdd.mapPartitions(apriori)
    candidates = candidates_rdd \
        .groupByKey() \
        .sortBy(lambda freq_set: freq_set[0]) \
        .collect()
    candidates_by_size = defaultdict(list)
    for size, candidate_item_sets in candidates:
        candidates_by_size[size] = sorted(set(reduce(lambda lis1, lis2: lis1 + lis2, candidate_item_sets, list())))
    # print the candidates
    write_item_sets_by_count(candidates_by_size, 'Candidates:', 'w')

    # find the truly frequent item sets - eliminate false positives - second phase
    frequents = list(filter(
        lambda item_set_count: item_set_count[1] >= params['support'],
        filtered_rdd.mapPartitions(lambda chunk: get_frequents(chunk, candidates_by_size)).reduceByKey(add).collect()
    ))
    frequents_by_size = defaultdict(list)
    for frequent_item_set, _ in frequents:
        if type(frequent_item_set) == tuple:
            frequents_by_size[len(frequent_item_set)].append(frequent_item_set)
        else:
            frequents_by_size[1].append(frequent_item_set)
    for frequent_item_set in frequents_by_size:
        frequents_by_size[frequent_item_set] = sorted(frequents_by_size[frequent_item_set])
    write_item_sets_by_count(frequents_by_size, 'Frequent Itemsets:', 'a')


# set executables
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# initialize program parameters
params = dict()
parse_args()

# create spark context
sc = SparkContext(conf=SparkConf().setAppName(params['app_name']).setMaster("local[*]"))
sc.setLogLevel('ERROR')

# Part A. parse raw data and generate the intermediate file
parse_raw_data()

# Part B. use the intermediate file, filter the counts and apply SON + Apriori
processed_rdd = parse_processed_data()
# filter the rdd tp get only those transactions that meet the filter threshold
filtered_rdd = filter_processed_data()
total_transaction_count = filtered_rdd.count()

# run SON with Apriori
start_ts = datetime.now()
execute_son()
end_ts = datetime.now()
print('Duration: ', (end_ts - start_ts).total_seconds())

# exit without errors
exit(0)
# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2.py 3 4 ../resource/asnlib/publicdata/ta_feng_all_months_merged.csv task2.json