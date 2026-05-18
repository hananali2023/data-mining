import sys
import time
import json
import numpy as np
import csv
import math
from pyspark import SparkContext, SparkConf
from xgboost import XGBRegressor



def review_features(review_rdd):
    def map_review(record):
        return (record['business_id'], (record['useful'], record['funny'], record['cool']))

    def reduce_review(values):
        useful_sum, funny_sum, cool_sum, count = 0, 0, 0, 0
        for u, f, c in values:
            useful_sum += u
            funny_sum += f
            cool_sum += c
            count += 1
        return (useful_sum / count, funny_sum / count, cool_sum / count)

    return review_rdd.map(map_review).groupByKey().mapValues(reduce_review).collectAsMap()



def user_features(user_rdd):
    def map_user(record):
        return (record['user_id'], (record['average_stars'], record['review_count'], record['fans']))

    return user_rdd.map(map_user).collectAsMap()



def business_features(business_rdd):
    def map_business(record):
        return (record['business_id'], (record['stars'], record['review_count']))

    return business_rdd.map(map_business).collectAsMap()



def create_feature_vectors(data_rdd, reviews_data, user_data, business_data, is_training=True):
    def map_features(record):
        user_id = record[0]
        business_id = record[1]
        rating = float(record[2]) if is_training else None

        review_feat = reviews_data.get(business_id, (0, 0, 0))
        user_feat = user_data.get(user_id, (3.0, 0, 0))
        business_feat = business_data.get(business_id, (3.0, 0))

        features = list(review_feat) + list(user_feat) + list(business_feat)
        return ((user_id, business_id), features, rating)

    return data_rdd.map(map_features)



def xgboost_model(feature_rdd):
    features = feature_rdd.map(lambda x: x[1]).collect()
    labels = feature_rdd.map(lambda x: x[2]).collect()

    X_train = np.array(features, dtype=np.float32)
    y_train = np.array(labels, dtype=np.float32)

    params = {
        'objective': 'reg:linear',
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100,
        'random_state': 42
    }

    xgb_model = XGBRegressor(**params)
    xgb_model.fit(X_train, y_train, verbose=1)
    return xgb_model


def predict_ratings(model, feature_rdd):
    user_business_pairs = feature_rdd.map(lambda x: x[0]).collect()
    features = feature_rdd.map(lambda x: x[1]).collect()

    X_test = np.array(features, dtype=np.float32)
    predictions = model.predict(X_test)

    return list(zip(user_business_pairs, predictions))



if __name__ == "__main__":
    train_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    start_time = time.time()
    conf = SparkConf().setAppName("task2_2")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    train_file = train_path + '/yelp_train.csv'
    train_rdd = sc.textFile(train_file)
    train_header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))

    test_rdd = sc.textFile(test_file)
    test_header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))

    reviews = train_path + '/review_train.json'
    user_info = train_path + '/user.json'
    business_info = train_path + '/business.json'

    review_rdd = sc.textFile(reviews).map(lambda x: json.loads(x))
    user_rdd = sc.textFile(user_info).map(lambda x: json.loads(x))
    business_rdd = sc.textFile(business_info).map(lambda x: json.loads(x))

    reviews_data = review_features(review_rdd)
    user_data = user_features(user_rdd)
    business_data = business_features(business_rdd)

    train_features_rdd = create_feature_vectors(train_rdd, reviews_data, user_data, business_data, is_training=True)


    model = xgboost_model(train_features_rdd)

    test_features_rdd = create_feature_vectors(test_rdd, reviews_data, user_data, business_data, is_training=False)


    predictions = predict_ratings(model, test_features_rdd)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['user_id', 'business_id', 'prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (user_id, business_id), pred in predictions:
            writer.writerow({'user_id': user_id, 'business_id': business_id, 'prediction': pred})


    real_ratings = '../resource/asnlib/publicdata/yelp_val.csv'
    real_rdd = sc.textFile(real_ratings)
    real_header = real_rdd.first()
    stars_rdd = (real_rdd.filter(lambda x: x != real_header).map(lambda x: x.split(',')).map(lambda x: ((x[0], x[1]), float(x[2]))))

    predictions_rdd = sc.parallelize(predictions)

    combined_rdd = predictions_rdd.join(stars_rdd)


    n = combined_rdd.count()
    squared_errors = combined_rdd.map(lambda x: (x[1][0] - x[1][1]) ** 2).sum()
    rmse_value = math.sqrt(squared_errors / n)


    sc.stop()
    end_time = time.time()

#  /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit task2_2.py ../resource/asnlib/publicdata ../resource/asnlib/publicdata/yelp_val.csv task2_2.csv