import sys
import time
import json
import csv
import numpy as np
from pyspark import SparkContext, SparkConf
from xgboost import XGBRegressor
import math



def item_similarity(item1, item2, item_user_stars):
    user1 = set(item_user_stars[item1].keys())
    user2 = set(item_user_stars[item2].keys())
    common_users = user1 & user2

    if len(common_users) < 2:
        avg1 = np.mean(list(item_user_stars[item1].values()))
        avg2 = np.mean(list(item_user_stars[item2].values()))
        return (5 - abs(avg1 - avg2)) / 5

    rating1 = np.array([item_user_stars[item1][n] for n in common_users])
    rating2 = np.array([item_user_stars[item2][n] for n in common_users])

    mean1 = np.mean(rating1)
    mean2 = np.mean(rating2)
    numerator = np.sum((rating1 - mean1) * (rating2 - mean2))
    denominator = np.sqrt(np.sum((rating1 - mean1)** 2)) * np.sqrt(np.sum((rating2 - mean2)**2))

    if denominator == 0:
        return 0
    return numerator / denominator


def predict_item(user, item, user_item, item_user_stars, item_similarities, default_stars=2.5):
    if user not in user_item or item not in item_user_stars:
        return default_stars

    user_items = user_item[user]
    similarities = []
    for others in user_items:
        pair = tuple(sorted([item, others]))
        if pair not in item_similarities:
            similar = item_similarity(item, others, item_user_stars)
            item_similarities[pair] = similar
        else:
            similar = item_similarities[pair]
        similarities.append((similar, item_user_stars[others][user]))

    if not similarities:
        return default_stars

    similarities = sorted(similarities, key=lambda x: -x[0])[:15]
    numerator = sum(sim * rating for sim, rating in similarities)
    denominator = sum(abs(sim) for sim, _ in similarities)

    if denominator == 0:
        return default_stars
    return numerator / denominator


def model_features(user_data, business_data, review_data, train_rdd, test_rdd):
    user_features = user_data.map(lambda x: (x['user_id'], [x['average_stars'], x['review_count'], x['fans']])).collectAsMap()

    business_features = business_data.map(lambda x: (x['business_id'], [x['stars'], x['review_count']])).collectAsMap()

    review_features = review_data.map(lambda x: (x['business_id'], [x['useful'], x['funny'], x['cool']])).groupByKey().mapValues(lambda vals: np.mean(list(vals), axis=0).tolist()).collectAsMap()


    def extract_train_features(row):
        user, business, rating = row
        user_feat = user_features.get(user, [0.0, 0.0, 0.0])
        business_feat = business_features.get(business, [0.0, 0.0])
        review_feat = review_features.get(business, [0.0, 0.0, 0.0])
        features = review_feat + user_feat + business_feat
        return features, float(rating)

    train_features = train_rdd.map(extract_train_features).collect()
    X_train = np.array([feat for feat, _ in train_features], dtype=np.float32)
    y_train = np.array([rating for _, rating in train_features], dtype=np.float32)

    def extract_test_features(row):
        user, business = row
        user_feat = user_features.get(user, [0.0, 0.0, 0.0])
        business_feat = business_features.get(business, [0.0, 0.0])
        review_feat = review_features.get(business, [0.0, 0.0, 0.0])
        features = review_feat + user_feat + business_feat
        return features

    test_user_business = test_rdd.map(lambda x: (x[0], x[1])).collect()
    X_test = np.array(test_rdd.map(extract_test_features).collect(), dtype=np.float32)

    return X_train, y_train, X_test, test_user_business




def model_based_predictions(X_train, y_train, X_test):
    xgb_model = XGBRegressor(
        objective = 'reg:linear',
        max_depth = 10,
        learning_rate = 0.05,
        n_estimators = 100,
        verbosity = 0,
        n_jobs = 1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_model = xgb_model.predict(X_test)
    return y_pred_model



def combined_predictions(test_user_business, item_predictions, model_preds, alpha=0.1):
    combined = []
    for i in range(len(test_user_business)):
        user, business = test_user_business[i]
        combined_ratings = alpha * item_predictions[i] + (1 - alpha) * model_preds[i]
        combined.append((user, business, combined_ratings))
    return combined



if __name__ == '__main__':
    folder_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_path = sys.argv[3]

    start_time = time.time()
    conf = SparkConf().setAppName("task2_3")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    train_rdd = sc.textFile(folder_path + "/yelp_train.csv")
    train_header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda line: line != train_header).map(lambda line: line.split(","))

    test_rdd = sc.textFile(test_file_path)
    test_header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda line: line != test_header).map(lambda line: line.split(",")).map(lambda arr: (arr[0], arr[1]))


    user_data = sc.textFile(folder_path + "/user.json")
    business_data = sc.textFile(folder_path + "/business.json")
    review_data = sc.textFile(folder_path + "/review_train.json")

    user_rdd = user_data.map(lambda line: json.loads(line))
    business_rdd = business_data.map(lambda line: json.loads(line))
    review_rdd = review_data.map(lambda line: json.loads(line))


    user_item = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    item_user_stars = train_rdd.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()

    item_similarities = {}
    predictions_rdd = test_rdd.map(lambda x: (x[0], x[1], predict_item(x[0], x[1], user_item, item_user_stars, item_similarities)))
    predictions = predictions_rdd.map(lambda x: x[2]).collect()

    X_train, y_train, X_test, test_user_business = model_features(user_rdd, business_rdd, review_rdd, train_rdd, test_rdd)


    model_predictions = model_based_predictions(X_train, y_train, X_test)

    final = combined_predictions(test_user_business, predictions, model_predictions, alpha=0.1)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'business_id', 'prediction'])
        writer.writerows(final)

    real_ratings = '../resource/asnlib/publicdata/yelp_val.csv'
    real_rdd = sc.textFile(real_ratings)
    real_header = real_rdd.first()
    stars_rdd = (real_rdd.filter(lambda x: x != real_header).map(lambda x: x.split(',')).map(lambda x: ((x[0], x[1]), float(x[2]))))

    predictions_rdd = sc.parallelize(final).map(lambda x: ((x[0], x[1]), float(x[2])))

    combined_rdd = predictions_rdd.join(stars_rdd)

    n = combined_rdd.count()
    squared_errors = combined_rdd.map(lambda x: (x[1][0] - x[1][1]) ** 2).sum()
    rmse_value = math.sqrt(squared_errors / n)


    sc.stop()



    #print(f"Duration: {time.time() - start_time}")
    #print(f"RMSE: {rmse_value}")

# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit task2_3.py ../resource/asnlib/publicdata ../resource/asnlib/publicdata/yelp_val.csv task2_3.csv