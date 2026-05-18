from pyspark import SparkContext
import sys
import time
import random
from itertools import combinations



def make_hash_functions(num_funcs, max_val):
    return [(random.randint(1, max_val), random.randint(1, max_val)) for _ in range(num_funcs)]


def create_signature_matrix(business_users, num_funcs, user_index, bins, prime, hash_funcs):
    signatures = {}
    for business, users in business_users:
        signature = [min(((a * user_index[user] + b) % prime) % bins for user in users)
                     for a, b in hash_funcs]
        signatures[business] = signature
    return signatures


def lsh(signatures, band_size):
    lsh_buckets = {}
    for business, signature in signatures.items():
        for i in range(0, len(signature), band_size):
            band = tuple(signature[i:i + band_size])
            lsh_buckets.setdefault(band, []).append(business)
    return lsh_buckets



def make_candidate_pairs(lsh_buckets):
    pairs = set()
    for bucket in lsh_buckets.values():
        if len(bucket) > 1:
            pairs.update(combinations(sorted(bucket), 2))
    return pairs




def jaccard_similarity(candidate_pairs, business_users, threshold):
    similar_pairs = {}
    for bus1, bus2 in candidate_pairs:
        if bus1 != bus2: # Latest change
            users1, users2 = business_users[bus1], business_users[bus2]
            jaccard_sim = len(users1 & users2) / len(users1 | users2)
            if jaccard_sim >= threshold:
                similar_pairs[(bus1, bus2)] = jaccard_sim
    return similar_pairs


def main(input_file, output_file):
    sc = SparkContext(appName="task1")
    start_time = time.time()

    data = sc.textFile(input_file).filter(lambda row: row != "user_id,business_id,stars").map(lambda x: x.split(","))

    all_users = data.map(lambda x: x[0])
    all_business = data.map(lambda x: x[1])

    business_users_rdd = data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)

    distinct_users = data.map(lambda x: x[0]).distinct().collect()

    user_index = {user: idx for idx, user in enumerate(distinct_users)}

    business_users_dict = business_users_rdd.collectAsMap()

    num_hashes = 60
    len_users = len(user_index)
    prime = 9876543210

    hash_funcs = make_hash_functions(num_hashes, len_users)
    #print(hash_funcs)

    signature_matrix = create_signature_matrix(business_users_dict.items(), num_hashes, user_index, len_users, prime, hash_funcs)

    band_size = 2
    buckets = lsh(signature_matrix, band_size)

    candidate_pairs = make_candidate_pairs(buckets)

    similarity_threshold = 0.5
    similar_pairs = jaccard_similarity(candidate_pairs, business_users_dict, similarity_threshold)

    with open(output_file, "w") as f:
        f.write("business_id_1,business_id_2,similarity\n")
        for (bus1, bus2), sim in sorted(similar_pairs.items()):
            f.write(f"{bus1},{bus2},{sim}\n")



if __name__ == '__main__':
    input_file, output_file = sys.argv[1], sys.argv[2]

    main(input_file, output_file)

# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit task1.py ../resource/asnlib/publicdata/yelp_train.csv task1.csv