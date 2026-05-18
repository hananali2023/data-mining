import sys
from pyspark import SparkContext, SparkConf
import time
import math

def pearson(business_pair, business_user_dict, user_avg_dict):
    bus1, bus2 = business_pair
    users_bus1 = business_user_dict.get(bus1, {})
    users_bus2 = business_user_dict.get(bus2, {})
    common_users = set(users_bus1.keys()).intersection(set(users_bus2.keys()))

    if len(common_users) == 0:
        return 0

    numerator, n1, n2 = 0, 0, 0
    for user in common_users:
        avg_rating = user_avg_dict.get(user, 2.5)
        r1 = users_bus1[user] - avg_rating
        r2 = users_bus2[user] - avg_rating
        numerator += r1 * r2
        n1 += r1**2
        n2 += r2**2

    if n1 == 0 or n2 == 0:
        return 0
    return numerator / (math.sqrt(n1) * math.sqrt(n2))

def predict_rating(user, business,
                   user_business_dict,
                   business_user_dict,
                   user_avg_dict,
                   global_avg,
                   business_sims_dict):
    if user not in user_business_dict:
        return 2.5
    if business not in business_user_dict:
        return user_avg_dict.get(user, global_avg)

    user_rated_businesses = user_business_dict[user]
    similarities = []
    for other_business, rating in user_rated_businesses.items():
        if other_business == business:
            continue
        pair = tuple(sorted((business, other_business)))


        if pair in business_sims_dict:
            sim = business_sims_dict[pair]
        else:
            sim = pearson(pair, business_user_dict, user_avg_dict)
            business_sims_dict[pair] = sim

        similarities.append((sim, rating))

    if not similarities:
        return 2.5

    similarities = sorted(similarities, key=lambda x: -abs(x[0]))[:15]
    numerator = sum(sim * rating for sim, rating in similarities)
    denominator = sum(abs(sim) for sim, _ in similarities)
    return 2.5 if denominator == 0 else numerator / denominator

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    start_time = time.time()

    conf = SparkConf().setAppName("task2_1")
    sc = SparkContext(conf=conf)


    train_rdd = sc.textFile(train_path) \
                  .filter(lambda row: row != "user_id,business_id,stars") \
                  .map(lambda row: row.split(",")) \
                  .map(lambda x: (x[0], x[1], float(x[2])))


    test_rdd = sc.textFile(test_path) \
                 .filter(lambda row: row != "user_id,business_id,stars") \
                 .map(lambda row: row.split(",")) \
                 .map(lambda x: (x[0], x[1]))


    global_avg = train_rdd.map(lambda x: x[2]).mean()
    user_business = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey() \
                             .mapValues(dict) \
                             .collectAsMap()
    business_user = train_rdd.map(lambda x: (x[1], (x[0], x[2]))) \
                             .groupByKey() \
                             .mapValues(dict) \
                             .collectAsMap()
    user_avg = train_rdd.map(lambda x: (x[0], x[2])) \
                        .groupByKey() \
                        .mapValues(lambda vals: sum(vals) / len(vals)) \
                        .collectAsMap()

    business_similarity_dict = {}

    user_business_bc = sc.broadcast(user_business)
    business_user_bc = sc.broadcast(business_user)
    user_avg_bc = sc.broadcast(user_avg)
    global_avg_bc = sc.broadcast(global_avg)

    business_similarity_bc = sc.broadcast(business_similarity_dict)

    def compute_prediction(x):
        user_id, business_id = x
        return (user_id, business_id, predict_rating( user_id, business_id, user_business_bc.value, business_user_bc.value, user_avg_bc.value, global_avg_bc.value, business_similarity_bc.value))

    predictions = test_rdd.map(compute_prediction).collect()

    with open(output_path, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for u, b, p in predictions:
            f.write(f"{u},{b},{p}\n")

    sc.stop()
    print("Execution time:", time.time() - start_time)
