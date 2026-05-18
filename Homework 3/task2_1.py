import sys
import time
from pyspark import SparkContext

def compute_similarity(item1, item2, item_user_dict, item_user_rating_dict, item_avg_rating, similarity):
    key = tuple(sorted((item1, item2)))
    if key in similarity:
        return similarity[key]

    users_item1 = item_user_dict.get(item1, set())
    users_item2 = item_user_dict.get(item2, set())
    common_users = users_item1 & users_item2

    if len(common_users) <= 1:
        pearson_similar = (5 - abs(item_avg_rating[item1] - item_avg_rating[item2])) / 5
    else:
        ratings_item1 = [item_user_rating_dict[item1][user] for user in common_users]
        ratings_item2 = [item_user_rating_dict[item2][user] for user in common_users]

        avg_item1 = sum(ratings_item1) / len(ratings_item1)
        avg_item2 = sum(ratings_item2) / len(ratings_item2)

        numerator = sum((r1 - avg_item1) * (r2 - avg_item2) for r1, r2 in zip(ratings_item1, ratings_item2))
        denominator = (sum((r - avg_item1) ** 2 for r in ratings_item1) ** 0.5) * (sum((r - avg_item2) ** 2 for r in ratings_item2) ** 0.5)

        pearson_similar = numerator / denominator if denominator != 0 else 0

    similarity[key] = pearson_similar
    return pearson_similar

def predict_rating(user, item, user_item_dict, item_user_dict, item_user_rating_dict, user_avg_rating, item_avg_rating, sim_cache):
    if user not in user_item_dict:
        return 2.5
    if item not in item_user_dict:
        return user_avg_rating.get(user, 2.5)

    similarities = []
    for rated_item in user_item_dict[user]:
        sim = compute_similarity(item, rated_item, item_user_dict, item_user_rating_dict, item_avg_rating, sim_cache)
        if sim > 0:
            rated_item_rating = item_user_rating_dict[rated_item][user]
            similarities.append((sim, rated_item_rating))

    if not similarities:
        return item_avg_rating.get(item, 2.5)


    top_similarities = sorted(similarities, key=lambda x: -x[0])[:15]
    numerator = sum(sim * rating for sim, rating in top_similarities)
    denominator = sum(abs(sim) for sim, _ in top_similarities)

    return numerator / denominator if denominator != 0 else 2.5

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]


    start_time = time.time()
    sc = SparkContext(appName="task2_1")

    train_data = sc.textFile(train_file).filter(lambda x: 'user_id' not in x).map(lambda x: x.split(","))
    user_item_ratings = train_data.map(lambda x: (x[0], x[1], float(x[2])))

    user_item_dict = user_item_ratings.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    item_user_dict = user_item_ratings.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
    item_user_rating_dict = user_item_ratings.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict).collectAsMap()

    user_avg_rating = user_item_ratings.map(lambda x: (x[0], x[2])).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()
    item_avg_rating = user_item_ratings.map(lambda x: (x[1], x[2])).groupByKey().mapValues(lambda x: sum(x)/len(x)).collectAsMap()


    test_data = sc.textFile(test_file).filter(lambda x: 'user_id' not in x).map(lambda x: x.split(","))
    test_pairs = test_data.map(lambda x: (x[0], x[1]))

    similarity = {}
    predictions = test_pairs.map(lambda x: (x[0], x[1], predict_rating(
        x[0], x[1], user_item_dict, item_user_dict, item_user_rating_dict, user_avg_rating, item_avg_rating, similarity
    ))).collect()

    with open(output_file, 'w') as f:
        f.write("user_id,business_id,prediction\n")
        for user_id, business_id, pred in predictions:
            f.write(f"{user_id},{business_id},{pred}\n")

    end_time = time.time()
    print("Duration:", end_time - start_time)


#  /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit task2_1.py ../resource/asnlib/publicdata/yelp_train.csv ../resource/asnlib/publicdata/yelp_val.csv task2_1.csv