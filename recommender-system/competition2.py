import os
import sys
import time
from pyspark import SparkContext
import numpy as np
import csv
import json
from xgboost import XGBRegressor
import random

def load_data(sc, path):
    return (sc.textFile(path).filter(lambda l: not l.startswith("user_id,")).map(lambda l: l.split(",")))


def all_mappings(train_rdd):
    user_to_business = (train_rdd.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap())
    business_to_user = (train_rdd.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap())
    business_avg = (train_rdd.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda ratings: sum(ratings) / len(ratings)).collectAsMap())
    user_avg = (train_rdd.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda ratings: sum(ratings) / len(ratings)).collectAsMap())

    return user_to_business, business_to_user, business_avg, user_avg


def load_business_json(sc, input_filepath):
     bus_data = sc.textFile(input_filepath).map(lambda x: json.loads(x))
     bus_attributes = bus_data.map(lambda x: (x["business_id"],
         (float(x.get("stars", 0)), float(x.get("review_count", 0)),
             int(x.get("attributes", {}).get("RestaurantsGoodForGroups", "False") == "True") if x.get("attributes") else 0,
             int(x.get("attributes", {}).get("GoodForKids", "False") == "True") if x.get("attributes") else 0,
             int(x.get("attributes", {}).get("BusinessAcceptsCreditCards", "False") == "True") if x.get("attributes") else 0,
             int(x.get("attributes", {}).get("RestaurantsDelivery", "False") == "True") if x.get("attributes") else 0,
             int(x.get("attributes", {}).get("RestaurantsReservations", "False") == "True") if x.get("attributes") else 0,
             int(x.get("attributes", {}).get("WheelchairAccessible", "False") == "True") if x.get("attributes") else 0,
             int(x.get("is_open", 0)),
             int(x.get("attributes", {}).get("ByAppointmentOnly", "False") == "True") if x.get("attributes") else 0,
             int(x.get("attributes", {}).get("OutdoorSeating", "False") == "True") if x.get("attributes") else 0,
             int(x.get("attributes", {}).get("BikeParking", "False") == "True") if x.get("attributes") else 0,
             int(x.get("attributes", {}).get("RestaurantsPriceRange2", "0")) if x.get("attributes") else random.randint(1, 4))))
     return bus_attributes


def load_jsons(sc, input_filepath):
     user_data = sc.textFile(input_filepath).map(lambda x: json.loads(x))
     user_attributes = user_data.map(lambda x: (x["user_id"],
         (float(x.get("review_count", 0)), float(x.get("useful", 0)), float(x.get("average_stars", 0)),
             int(len(x.get("elite", "").split(',')) > 0) if x.get("elite") else 0, int(x.get("fans", 0)),
             int(x.get("funny", 0)), int(x.get("cool", 0)), int(x.get("compliment_hot", 0)),
             int(x.get("compliment_cool", 0)), int(x.get("compliment_funny", 0)), int(x.get("compliment_photos", 0)))))
     return user_attributes



def load_checkin_json(sc, input_filepath):
     checkin_data = sc.textFile(input_filepath).map(lambda x: json.loads(x))
     checkin_attributes = checkin_data.map(lambda x: (x["business_id"], int(sum(x.get("time", {}).values()))))
     return checkin_attributes


def process_train(row, bus_dict, usr_dict, checkin_dict):
     usr, bus = row[:2]
     rating = row[2] if len(row) > 2 else None
     bus_features = bus_dict.get(bus, (None,) * 14)
     user_features = usr_dict.get(usr, (None,) * 11)
     checkins = checkin_dict.get(bus, None)
     return ([*user_features, *bus_features, checkins], rating)


def pearson(b1, b2, business_user, business_avg):
    user1 = set(business_user.get(b1, {}).keys())
    user2 = set(business_user.get(b2, {}).keys())
    share = user1 & user2

    if len(share) <= 1:
        return 1 - abs(business_avg.get(b1, 0.0) - business_avg.get(b2, 0.0)) / 5.0

    r1 = [business_user[b1][u] for u in share]
    r2 = [business_user[b2][u] for u in share]
    m1, m2 = sum(r1)/len(r1), sum(r2)/len(r2)

    num = sum((x-m1)*(y-m2) for x,y in zip(r1,r2))
    den = (sum((x-m1)**2 for x in r1)*sum((y-m2)**2 for y in r2))**0.5
    return num/den if den != 0 else 0.0

def predict(user, business, user_to_business, business_to_user, business_avg, k=15):
    if user not in user_to_business or business not in business_to_user:
        return 2.5
    similar = []
    for other_bus, rating in user_to_business[user].items():
        if other_bus == business: continue
        similar_score = pearson(business, other_bus, business_to_user, business_avg)
        similar.append((similar_score, rating))

    top = sorted(similar, key = lambda x: -x[0])[:k]
    if not top:
        return 2.5
    num = sum(similar * r for similar, r in top)
    den = sum(abs(similar) for similar, _ in top)
    return num / den if den != 0 else 2.5

# Main function with XGBoost

def main(folder_path, test_filepath, output_filepath):
    sc = SparkContext("local[*]", appName="competition")
    start_time = time.time()


    yelp_train = os.path.join(folder_path, "yelp_train.csv")
    business_json = os.path.join(folder_path, "business.json")
    user_json = os.path.join(folder_path, "user.json")
    checkin_json = os.path.join(folder_path, "checkin.json")


    train_rdd = load_data(sc, yelp_train)
    test_rdd = load_data(sc, test_filepath)

    user_to_business, business_to_user, business_avg, user_avg = all_mappings(train_rdd)

    coll_fil = test_rdd.map(lambda x: ((x[0], x[1]), predict(x[0], x[1], user_to_business, business_to_user, business_avg))).collectAsMap()



    business_rdd = load_business_json(sc, business_json)
    user_rdd = load_jsons(sc, user_json)
    checkin_rdd = load_checkin_json(sc, checkin_json)


    bus_dict = business_rdd.collectAsMap()
    user_dict = user_rdd.collectAsMap()
    checkin_dict = checkin_rdd.collectAsMap()

    train_processed = train_rdd.map(lambda x: process_train(x, bus_dict, user_dict, checkin_dict))
    X_train = np.array(train_processed.map(lambda x: x[0]).collect(), dtype="float32")
    Y_train = np.array(train_processed.map(lambda x: x[1]).collect(), dtype="float32")

    validation_data = test_rdd.map(lambda x: process_train(x, bus_dict, user_dict, checkin_dict))
    x_validation = np.array(validation_data.map(lambda x: x[0]).collect(), dtype="float32")

    # XGBoost model
    xgboost = XGBRegressor(
        subsample=0.6,reg_lambda=1.5, reg_alpha=0,
        n_estimators=300, max_depth=7, learning_rate=0.05,
        colsample_bytree=0.6, tree_method='hist', random_state=42)

    xgboost.fit(X_train, Y_train)
    y_pred = xgboost.predict(x_validation)

    actual_ratings = np.array(test_rdd.map(lambda x: float(x[2])).collect(), dtype="float32")


    # Calculate error distributions
    absolute_errors = np.abs(y_pred - actual_ratings)

    error_distribution = {
        ">=0 and <1": 0,
        ">=1 and <2": 0,
        ">=2 and <3": 0,
        ">=3 and <4": 0,
        ">=4": 0
    }

    for error in absolute_errors:
        if 0 <= error < 1:
            error_distribution[">=0 and <1"] += 1
        elif 1 <= error < 2:
            error_distribution[">=1 and <2"] += 1
        elif 2 <= error < 3:
            error_distribution[">=2 and <3"] += 1
        elif 3 <= error < 4:
            error_distribution[">=3 and <4"] += 1
        elif error >= 4:
            error_distribution[">=4"] += 1

    print("Error Distribution:")
    for rng, count in error_distribution.items():
        print(f"{rng}: {count}")

    with open(output_filepath, "w+") as f:
        f.write("user_id,business_id,prediction\n")
        for i, row in enumerate(test_rdd.collect()):
            f.write(f"{row[0]},{row[1]},{y_pred[i]}\n")


    # # RSME
    # rmse = np.sqrt(np.mean((y_pred - actual_ratings) ** 2))
    # print(f"RMSE: {rmse}")
    #
    # # Duration
    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"Duration: {duration} seconds")

    sc.stop()

if __name__ == "__main__":
    folder_path, test_csv, output_filepath = sys.argv[1], sys.argv[2], sys.argv[3]
    main(folder_path, test_csv, output_filepath)

# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit competition.py ../resource/asnlib/publicdata/ ../resource/asnlib/publicdata/yelp_val.csv output.txt