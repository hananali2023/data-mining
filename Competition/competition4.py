import os
import sys
import time
from pyspark import SparkContext
import numpy as np
import csv
import json
from xgboost import XGBRegressor


# First part of the code will be implementing Item Based CF.

"""
Defining a function to load the data into RDD's. Included a split argument to check if we are loading the training or testing dataset.
If no argument for split is specified, it will by default split it according to the train criteria. 

Returns a RDD of the training/testing set. 
"""


def load_data(sc, train_filepath, split="train"):
    data = sc.textFile(train_filepath)

    # Filter out the header
    header = data.first()
    if split == "train":
        train_rdd = data.filter(lambda x: x != header).map(lambda x: x.split(","))
        return train_rdd
        # print(train_rdd.take(5))
    else:
        test_rdd = data.filter(lambda l: l != header).map(lambda l: l.split(","))

        return test_rdd
        # print(test_rdd.take(5))


"""
This function takes in the RDD from load_dataset and generates a dictionary mapping with the following format: 
user_id: {business_id, rating}. It then returns the dictionary as the output. 
"""


def user_to_business_mapping(train_rdd):
    # Ratings in the train_rdd is a string so need to convert it to a float before mapping
    user_business_dict = train_rdd.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(
        dict).collectAsMap()

    # Return the dictionary of mapped users to business
    return user_business_dict
    # print(dict(list(user_business_dict.items())[:1])) #Comment out before submitting, local test only


"""
This function takes in the RDD from load_dataset and generates a dictionary mapping with the following format: 
business_id: {user_id, rating}. It then returns the dictionary as the output. 
"""


def business_to_user_mapping(train_rdd):
    # Rating in the train_rdd is a string so need to convert it to a float before mapping
    business_user_dict = train_rdd.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(
        dict).collectAsMap()

    # Return the dictionary
    return business_user_dict
    # print(dict(list(business_user_dict.items())[:1])) #Comment out before submitting, local test only


"""
This function takes in the RDD from load_dataset and generates a dictionary of 
user_id: average_rating. Basically takes all the ratings of that user and returns an average rating
as the value. 
"""


def user_rating_avg(train_rdd):
    # Convert ratings to float and map them out, excluding x[1] here
    user_rating_avg_dict = train_rdd.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(
        lambda x: sum(x) / len(x)).collectAsMap()

    # Return the dictionary
    return user_rating_avg_dict
    # print(dict(list(user_rating_avg_dict.items())[:1])) #Comment out before submitting, local test only


"""
This function takes in the RDD from load_dataset and generates a dictionary of 
business_id: average_rating. Basically takes all the ratings of that business and returns an average rating
as the value. 
"""


def business_rating_avg(train_rdd):
    # Convert ratings to float and map them out, this time excluding x[0]
    business_rating_avg_dict = train_rdd.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(
        lambda x: sum(x) / len(x)).collectAsMap()

    # Return the dictionary
    return business_rating_avg_dict
    # print(dict(list(business_rating_avg_dict.items())[:1])) #Comment out before submitting, local test only


"""
Defining a function to calculate the pearson similarity for the items.
Inputs: The pair of items (businesses) that we want to calculate & the business_id: {user_id} dictionary from business_to_user_mapping.
Outputs: Pearson similarity value or if either numerator or denominator is 0, return an arbritrary 0. 
"""


def compute_pearson_similarity(business_pair, business_user_dict):
    business1, business2 = business_pair
    users_business1 = set(business_user_dict[business1].keys())
    users_business2 = set(business_user_dict[business2].keys())
    shared_users = users_business1 & users_business2

    if len(shared_users) <= 1:
        # Edge case if there is insufficient data or missing values from either businesses. 
        avg_diff = abs(sum(business_user_dict[business1].values()) / len(business_user_dict[business1]) -
                       sum(business_user_dict[business2].values()) / len(business_user_dict[business2]))
        return 1 - (avg_diff / 5)

    ratings1 = [business_user_dict[business1][user] for user in shared_users]
    ratings2 = [business_user_dict[business2][user] for user in shared_users]

    avg1 = sum(ratings1) / len(ratings1)
    avg2 = sum(ratings2) / len(ratings2)

    numerator = sum((r1 - avg1) * (r2 - avg2) for r1, r2 in zip(ratings1, ratings2))
    denominator = (sum((r1 - avg1) ** 2 for r1 in ratings1) * sum((r2 - avg2) ** 2 for r2 in ratings2)) ** 0.5

    return numerator / denominator if denominator != 0 else 0


"""
Defining a function to predict the rating on the testing data using the Pearson similarity values. 
We'll be using 2.5 as a rating if either the user or business is missing from the training data. 2.5 is just the average of the 
minimum and maximum possible rating (0.0 + 5.0)/2. 
"""


def predict_rating(user_business_pair, user_business_dict, business_user_dict, n_neighbors=15):
    user, business = user_business_pair

    if user not in user_business_dict or business not in business_user_dict:
        return 2.5  # Default rating changed to 2.5, had better performance with 2.5 than 2.75

    similarities = []
    for rated_business in user_business_dict[user]:
        if rated_business != business:
            sim = compute_pearson_similarity((business, rated_business), business_user_dict)
            similarities.append((sim, user_business_dict[user][rated_business]))

    top_similarities = sorted(similarities, key=lambda x: -x[0])[:n_neighbors]

    if not top_similarities:
        return 2.5  # Default rating if no similarities found, using just a arbitrary mean number of min and max ratings

    weighted_sum = sum(sim * rating for sim, rating in top_similarities)
    sum_of_similarities = sum(abs(sim) for sim, _ in top_similarities)

    return weighted_sum / sum_of_similarities if sum_of_similarities != 0 else 2.5


####################################################################################################################################################################
# Second part of the code will be implementing XGBoost Regressor to get the rating predictions. 

"""
Defining a function to load specifically the business.json file into a RDD. 
Attributes that I'm keeping from the file: business_id, stars and review_count. 
"""


def load_business_json(sc, input_filepath):
    # Read the file into a RDD
    bus_data = sc.textFile(input_filepath).map(lambda x: json.loads(x))

    # Map out only the attributes that will be used, make sure stars and review_counts are float values, used for calc later
    bus_attributes = bus_data.map(lambda x: (x["business_id"], (float(x["stars"]), float(x["review_count"]))))

    # Return the RDD
    return bus_attributes
    # print(bus_attributes.take(5)) # Comment out later, only for local testing


"""
Defining a function to load specifically the user.json file into a RDD.
Attributes that I'm keeping are the user_id, review_count, useful & average_stars.
"""


def load_user_json(sc, input_filepath):
    # Read the file into a RDD
    user_data = sc.textFile(input_filepath).map(lambda x: json.loads(x))

    # Map out only the attributes that will be used, make sure review_count, useful and average stars are all float values, used for calc later
    user_attributes = user_data.map(
        lambda x: (x["user_id"], (float(x["review_count"]), float(x["useful"]), float(x["average_stars"]))))

    # Return the RDD
    return user_attributes
    # print(user_attributes.take(5)) # Comment out later, only for local testing


"""
Defining a function to process the feature vectors as training data for the XGBoost model. 
When unpacking the features, we use none as placeholder default values if there aren't corresponding entries in the dictionary. 
https://stackoverflow.com/questions/7082966/should-i-return-none-or-none-none

"""


def process_train_data(row, bus_dict, usr_dict):
    # First, we extract the user & business_id from the row (assumed to be the first 2 elements)
    usr, bus = row[:2]

    # Then check if there is a third entry, if yes then set it = to rating, if not rating is set to default as None
    rating = row[2] if len(row) > 2 else None

    # Extract features from dictionaries, using None as the default placeholder values if the entries cannot be found
    bus_features = bus_dict.get(bus, (None, None))
    user_features = usr_dict.get(usr, (None, None, None))

    # Returns unpacked tuple values using the * for the sequence and then combine them all into one list
    # https://stackoverflow.com/questions/62671692/how-to-unpack-a-single-variable-tuple-in-python3
    return ([*user_features, *bus_features], rating)


"""
Defining the main function which will call all the previous sub functions. Key point here is that this function will read in the
intermediate results from the item based CF & the results from the XGBoost Regressor, then implement Method 1 in the handout. 
I will be playing around with the different weighted values to test for the best RMSE and then output the results of that combination. 
"""


def competition(folder_path, test_filepath, output_filepath):
    # Put everything in a try, finally block
    try:
        sc = SparkContext("local[*]", appName="competition")
        sc.setLogLevel("ERROR")

        # Start the timer
        start_time = time.time()

        """
        Getting the item based CF results first: 
        """
        # print("Running CF")
        # Define paths to the files that will be used in both item based & XGB, assuming same working directory in VC
        yelp_train = os.path.join(folder_path, "yelp_train.csv")
        business_json = os.path.join(folder_path, "business.json")
        user_json = os.path.join(folder_path, "user.json")

        # Load the dataset into RDD for first task
        train_rdd = load_data(sc, yelp_train, split="train")
        test_rdd = load_data(sc, test_filepath, split="test")

        # Create the business-user & user-business mappings & the average business & user ratings
        user_business_dict = user_to_business_mapping(train_rdd)
        business_user_dict = business_to_user_mapping(train_rdd)
        user_avg_ratings = user_rating_avg(train_rdd)
        business_avg_ratings = business_rating_avg(train_rdd)

        # Prepare the predictions
        predictions = test_rdd.map(lambda x: (x[0], x[1], predict_rating(x, user_business_dict, business_user_dict)))

        # Write the results to a temp file in the same directory
        temp_results1 = "./temp_results1.csv"
        with open(temp_results1, "w+") as f:
            f.write("user_id,business_id,prediction\n")
            for user_id, business_id, prediction in predictions.collect():
                f.write(f"{user_id},{business_id},{prediction}\n")

        # print("CF complete, results written out to file.")

        # print("Starting Model Based")
        """
        Model based prediction: XGBoost Regressor
        """
        # training & testing data already loaded into RDD, now load business.json & user.json
        bus_rdd = load_business_json(sc, business_json)
        user_rdd = load_user_json(sc, user_json)

        # Transform the business and user RDD's into dictionaries
        bus_dict = bus_rdd.collectAsMap()
        user_dict = user_rdd.collectAsMap()

        # Prepare the training data for XGB
        train_processed = train_rdd.map(lambda x: process_train_data(x, bus_dict, user_dict))
        X_train = np.array(train_processed.map(lambda x: x[0]).collect(), dtype="float32")
        Y_train = np.array(train_processed.map(lambda x: x[1]).collect(), dtype="float32")

        # Prepare validation data
        val_processed = test_rdd.map(lambda x: process_train_data(x, bus_dict, user_dict))
        X_val = np.array(val_processed.map(lambda x: x[0]).collect(), dtype="float32")

        # Train and predict using XGBoost with specific parameters
        xgb = XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=110)
        xgb.fit(X_train, Y_train)
        Y_pred = xgb.predict(X_val)
        actual_ratings = np.array(test_rdd.map(lambda x: float(x[2])).collect(), dtype="float32")

        # Write the predictions to a temporary csv file 
        temp_results2 = "./temp_results2.csv"
        with open(temp_results2, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "business_id", "prediction"])
            for i, row in enumerate(test_rdd.collect()):
                writer.writerow([row[0], row[1], Y_pred[i]])

        # print("Model based complete, written out to file.")

        """
        The following code block will:
        1. Read temp_results1 & temp_results2
        2. Implement this formula 𝑓𝑖𝑛𝑎𝑙 𝑠𝑐𝑜𝑟𝑒 = 𝛼 × 𝑠𝑐𝑜𝑟𝑒 (𝑖𝑡𝑒𝑚 𝑏𝑎𝑠𝑒𝑑) + (1 − 𝛼) × 𝑠𝑐𝑜𝑟𝑒 (𝑚𝑜𝑑𝑒𝑙 𝑏𝑎𝑠𝑒𝑑) based on each file's predictions
        3. Compile and write the final predictions to our final results at output_filepath
        """

        # Start off with 0.5 & 0.5 for the alpha values to test the RMSE scores and adjust from there 

        """ TOO TIME CONSUMING, EXCEEDED 1800S
        print("Starting extraction.")
        # Dictionaries to store the values from both the intermediate result files
        item_based_prediction = {}
        model_based_prediction = {}

        # Read the results from temp_results1
        with open(temp_results1, "r") as file1: 
            reader = csv.reader(file1)

            # Skip the first line
            next(reader)

            for row in reader:
                user_id, business_id, prediction = row

                # Append to dictionary, {(user_id, business_id): prediction}
                item_based_prediction[(user_id, business_id)] = float(prediction)


        # Read the results from temp_results2
        with open(temp_results2, "r") as file2:
            reader = csv.reader(file2)

            # Skip the first line
            next(reader)

            for row in reader:
                user_id, business_id, prediction = row

                # Append to dictionary, {(user_id, business_id): prediction}
                model_based_prediction[(user_id, business_id)] = float(prediction)


        print("Extraction complete, writing out to file now.")
        # Calculate the final score & write it to the output_filepath
        with open(output_filepath, "w+") as f: 
            # Write the header
            f.write("user_id,business_id,prediction\n")

            # Join both the 2 dictionaries by running 2 for loops and calculate the final score
            alpha = 0.5 # start off with 0.5 and adjust as we go
            for key in item_based_prediction:
                for key in model_based_prediction:
                    item_prediction = item_based_prediction[key]
                    model_prediction = model_based_prediction[key]

                    # Insert formula
                    hybrid_prediction = float(alpha * item_prediction + (1 - alpha) * model_prediction)
                    user_id, business_id = key
                    f.write(f"{user_id},{business_id},{hybrid_prediction}\n")
            """

        # print("Reading results into RDD")
        # Read both temp_results1 & temp_results2 into RDD's to process more efficiently
        item_based_rdd = sc.textFile(temp_results1).filter(lambda x: x != "user_id,business_id,prediction").map(
            lambda row: row.split(","))
        item_based_rdd = item_based_rdd.map(lambda x: ((x[0], x[1]), float(x[2])))

        model_based_rdd = sc.textFile(temp_results2).filter(lambda x: x != "user_id,business_id,prediction").map(
            lambda row: row.split(","))
        model_based_rdd = model_based_rdd.map(lambda x: ((x[0], x[1]), float(x[2])))

        # print("Joining both RDD's and applying formula")
        # Join both RDD's on user_id & business_id, apply formula on both their predictions
        alpha = 0.1
        hybrid_rdd = item_based_rdd.join(model_based_rdd).map(
            lambda x: (x[0][0], x[0][1], alpha * x[1][0] + (1 - alpha) * x[1][1]))

        rmse = np.sqrt(np.mean((Y_pred - actual_ratings) ** 2))
        print(f"RMSE: {rmse}")


        # print("Writing out results")
        # Write out the results to output_filepath
        with open(output_filepath, "w+") as f:
            f.write("user_id,business_id,prediction\n")
            for user_id, business_id, prediction in hybrid_rdd.collect():
                f.write(f"{user_id},{business_id},{prediction}\n")

        # Stop the timer and print duration to terminal
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration: {duration}")

    # Stop the SC 
    finally:
        sc.stop()

if __name__ == "__main__":
    folder_path = sys.argv[1]
    test_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    competition(folder_path, test_filepath, output_filepath)