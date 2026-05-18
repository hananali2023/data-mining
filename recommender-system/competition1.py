import sys
import time
import json
import os
import numpy as np
from pyspark import SparkConf, SparkContext
from xgboost import XGBRegressor
from datetime import datetime



# Method Description:
# This code implements a recommendation system for Yelp data using Spark for data processing and XGBoost for regression
# modeling. The code processes the training and testing Yelp data together containing reviews, user, business, tip and
# check-in data. I extracted features from these sources, such as review metrics (useful, funny, cool) like in
# Assignment 3, check-in metrics to generate a business popularity, user statistics such as average_stars and
# reveiw_count, business statistics about average_stars and review_count, and finally photo and whether a business has a
# photo. I tried a combination of these features and combined them into vectors for training and testing. I continued to
# add features from the json files provided to lower the RSME. Features such as Bike Parking from business.json and years
# users have been Yelping in user.json. Then, I ran an XGBoost regression model to train the features to predict user
# ratings for businesses. The predictions are generated for test data and evaluated against actual ratings from the
# validation dataset using Root Mean Square Error (RMSE).


# Error Distribution: {'0-1': 102004,
# '1-2': 33090,
# '2-3': 6154,
# '3-4': 796,
# '4-inf': 0}


# RMSE:
# 0.9801743843120851


# Execution Time:
# 333s



def spark_configuration():
    conf = SparkConf().setAppName("competition")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    return sc



def load_data(sc, folder_path):
    review_rdd = sc.textFile(f"{folder_path}/review_train.json").map(lambda x: json.loads(x))
    user_rdd = sc.textFile(f"{folder_path}/user.json").map(lambda x: json.loads(x))
    business_rdd = sc.textFile(f"{folder_path}/business.json").map(lambda x: json.loads(x))
    checkin_rdd = sc.textFile(f"{folder_path}/checkin.json").map(lambda x: json.loads(x))
    tip_rdd = sc.textFile(f"{folder_path}/tip.json").map(lambda x: json.loads(x))
    photo_rdd = sc.textFile(f"{folder_path}/photo.json").map(lambda x: json.loads(x))
    user_tip_counts = tip_rdd.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda a, b: a + b)
    business_text_review_counts = review_rdd.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda a, b: a + b)
    user_text_review_counts = review_rdd.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda a, b: a + b)

    return (review_rdd, user_rdd, business_rdd, checkin_rdd, photo_rdd, tip_rdd,
            user_tip_counts, business_text_review_counts, user_text_review_counts)



def extract_features(user_rdd, business_rdd, user_tip_counts, business_text_review_counts, user_text_review_counts):
    current_year = datetime.now().year
    user_features = user_rdd.map(lambda u: (
        u['user_id'],
        (u['review_count'],
            u['average_stars'],
            u['fans'],
            current_year - int(u['yelping_since'].split('-')[0]),
            1 if u['elite'] != "None" else 0,
            u['useful'] + u['funny'] + u['cool'], sum([u['compliment_hot'], u['compliment_cool'], u['compliment_more'], u['compliment_writer'], u['compliment_photos']]))))

    user_features = user_features.leftOuterJoin(user_tip_counts).mapValues(lambda x: x[0] + (x[1] or 0,))
    user_features = user_features.leftOuterJoin(user_text_review_counts).mapValues(lambda x: x[0] + (x[1] or 0,))



    business_features = business_rdd.map(lambda b: (b['business_id'],(b['review_count'],
        b['stars'],
        int(b.get('RestaurantsPriceRange2', 0)),
        1 if b.get('RestaurantsReservations', 'False') == 'True' else 0,
        1 if b.get('RestaurantsTakeOut', 'False') == 'True' else 0)))


    business_features = business_features.leftOuterJoin(business_text_review_counts).mapValues(lambda x: x[0] + (x[1] or 0,))


    return user_features, business_features


def extract_review_features(review_rdd):
    review_features = review_rdd.map(lambda r: (r['review_id'], len(r['text'].split())))
    review_features = (review_rdd.map(lambda x: (x['business_id'], [x['useful'], x['funny'], x['cool']])).groupByKey().mapValues(lambda vals: np.mean(list(vals), axis=0).tolist()))
    return review_features.collectAsMap()


# use checkin.json instead of the "cool, funny, useful" words from HW3
# can see business popularity and see if that has anything to do with their reviews!
def extract_checkin_features(checkin_rdd):
    checkin_features = checkin_rdd.map(lambda c: (c['business_id'], sum(c['time'].values())))
    return checkin_features

def extract_photo_features(photo_rdd):
    photo_features = photo_rdd.map(lambda p: (p['business_id'], 1)).reduceByKey(lambda a, b: a + b)
    return photo_features


def create_feature_vectors(data_rdd, user_features, business_features, checkin_features, photo_features):
    def map_features(row):
        try:
            user_id, business_id, stars = row[0], row[1], float(row[2])

            user_data = user_features.get(user_id, (0, 3, 0, 0, 0, 0, 0, 0))
            business_data = business_features.get(business_id, (0, 3, 0, 0, 0, 0))
            checkin_data = checkin_features.get(business_id, 0)
            photo_data = photo_features.get(business_id, 0)

            features = list(user_data) + list(business_data) + [checkin_data] + [photo_data]

            return ((user_id, business_id), features, stars)
        except Exception as e:
            print(f"Error processing row: {row}, Error: {e}")
            return None

    processed_rdd = data_rdd.map(map_features)
    return processed_rdd.filter(lambda x: x is not None)



def normalized(features):
    features = np.array(features)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / (stds + 1e-8)
    return normalized_features, means, stds




#os.environ['TMPDIR'] = '/mnt/vocwork/'
def train_xgboost(train_features_rdd):
    features = train_features_rdd.map(lambda x: x[1]).collect()
    labels = train_features_rdd.map(lambda x: x[2]).collect()

    features = np.array(features)
    labels = np.array(labels)

    # print(f"Features Shape: {features.shape}")
    # print(f"Labels Shape: {labels.shape}")

    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Training data (features or labels) is empty!")

    if len(features) != len(labels):
        raise ValueError(f"Mismatch: Features length ({len(features)}) != Labels length ({len(labels)})")

    if np.isnan(features).any() or np.isnan(labels).any():
        raise ValueError("Features or labels contain NaN values!")

    if np.isinf(features).any() or np.isinf(labels).any():
        raise ValueError("Features or labels contain infinite values!")

    params = {
        'objective': 'reg:linear',
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 200,
        'random_state': 42
    }
    xgb = XGBRegressor(**params)
    xgb.fit(features, labels, verbose=1)
    #xgb.save_model('/mnt/vocwork/xgboost_model.json')
    return xgb



def predict(model, test_features_rdd):
    features = test_features_rdd.map(lambda x: x[1]).collect()
    user_business_pairs = test_features_rdd.map(lambda x: x[0]).collect()
    predictions = model.predict(np.array(features))
    return list(zip(user_business_pairs, predictions))



def calculate_rmse(predictions_rdd, ground_truth_rdd):
    combined_rdd = predictions_rdd.join(ground_truth_rdd)
    squared_errors = combined_rdd.map(lambda x: (x[1][0] - x[1][1]) ** 2).sum()
    n = combined_rdd.count()
    return (squared_errors / n) ** 0.5


def error_distribution(predictions_rdd, ground_truth_rdd):
    combined_rdd = predictions_rdd.join(ground_truth_rdd)
    errors = combined_rdd.map(lambda x: abs(x[1][0] - x[1][1]))
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, float('inf'))]
    distribution = {f"{low}-{high}": errors.filter(lambda e: low <= e < high).count() for low, high in bins}
    return distribution


if __name__ == "__main__":
    start_time = time.time()


    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]


    sc = spark_configuration()


    #review_rdd, user_rdd, business_rdd, checkin_rdd, photo_rdd = load_data(sc, folder_path)
    (review_rdd, user_rdd, business_rdd, checkin_rdd, photo_rdd, tip_rdd, user_tip_counts, business_text_review_counts, user_text_review_counts) = load_data(sc, folder_path)


    # train rdd
    train_rdd = sc.textFile(f"{folder_path}/yelp_train.csv")
    train_header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))

    # test rdd
    test_rdd = sc.textFile(test_file_name)
    test_header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))

    # features
    #user_features = extract_features(user_rdd, business_rdd)[0].collectAsMap()
    #business_features = extract_features(user_rdd, business_rdd)[1].collectAsMap()

    user_features, business_features = extract_features(user_rdd, business_rdd, user_tip_counts, business_text_review_counts, user_text_review_counts)
    user_features = user_features.collectAsMap()
    business_features = business_features.collectAsMap()
    checkin_features = extract_checkin_features(checkin_rdd).collectAsMap()
    photo_features = extract_photo_features(photo_rdd).collectAsMap()
    review_features = extract_review_features(review_rdd)


    train_features_rdd = create_feature_vectors(train_rdd, user_features, business_features, checkin_features, photo_features)

    features = np.array(train_features_rdd.map(lambda x: x[1]).collect())

    # try normalizing for features to help
    normalized_features, means, stds = normalized(features)

    train_features_rdd = sc.parallelize([(row[0], normalized_features[i], row[2]) for i, row in enumerate(train_features_rdd.collect())])

    model = train_xgboost(train_features_rdd)


    test_features_rdd = create_feature_vectors(test_rdd, user_features, business_features, checkin_features, photo_features)

    test_features = np.array(test_features_rdd.map(lambda x: x[1]).collect())

    normalized_test_features = (test_features - means) / (stds + 1e-8)

    test_features_rdd = sc.parallelize([(row[0], normalized_test_features[i], row[2]) for i, row in enumerate(test_features_rdd.collect())])

    predictions = predict(model, test_features_rdd)
    predictions_rdd = sc.parallelize(predictions)

    ground_truth_rdd = sc.textFile(f"{folder_path}/yelp_val.csv")
    header = ground_truth_rdd.first()
    ground_truth_rdd = ground_truth_rdd.filter(lambda x: x != header).map(lambda x: x.split(',')).map(lambda x: ((x[0], x[1]), float(x[2])))

    rmse = calculate_rmse(predictions_rdd, ground_truth_rdd)
    error_dist = error_distribution(predictions_rdd, ground_truth_rdd)
    # print(f"RSME: {rmse}")
    # print(f"Duration: {time.time() - start_time}")
    # print(f"Error Distribution: {error_dist}")


    with open(output_file_name, 'w') as f:
        f.write("user_id,business_id,prediction\n")
        for (user_id, business_id), prediction in predictions:
            f.write(f"{user_id},{business_id},{prediction}\n")


    # with open("description.txt", 'w') as desc_file:
    #     desc_file.write(f"RMSE: {rmse}\n")
    #     desc_file.write(f"Execution Time: {time.time() - start_time}\n")
    #     desc_file.write(f"Error Distribution: {error_dist}\n")

    sc.stop()
