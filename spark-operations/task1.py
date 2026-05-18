
import json
import sys
from pyspark import SparkConf, SparkContext


def task1(input_file, output_file):
    # Spark Configuration
    conf = SparkConf().setAppName("task1")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")


    # Reviews RDD and parsed data
    reviews = sc.textFile(input_file)
    cleaned_reviews = reviews.map(lambda line: json.loads(line))
    total_reviews = cleaned_reviews.count()
    yr_2018_reviews = cleaned_reviews.filter(lambda review: review['date'].startswith('2018')).count()

    # Users and Top 10 Users
    all_users = cleaned_reviews.map(lambda review: review['user_id']).distinct().count()
    users = cleaned_reviews.map(lambda review: (review['user_id'], 1)).reduceByKey(lambda x, y: x + y)
    top_10_users = users.takeOrdered(10, key=lambda x: (-x[1], x[0]))

    # Businesses and Top 10 Businesses
    all_businesses = cleaned_reviews.map(lambda review: review['business_id']).distinct().count()
    businesses = cleaned_reviews.map(lambda review: (review['business_id'], 1)).reduceByKey(lambda x, y: x + y)
    top_10_businesses = businesses.takeOrdered(10, key=lambda x: (-x[1], x[0]))

    # Results!
    final_results = {"n_review": total_reviews,"n_review_2018": yr_2018_reviews,"n_user": all_users,
        "top10_user": top_10_users,"n_business": all_businesses,"top10_business": top_10_businesses}

    # Write results to output file
    with open(output_file, 'w') as output_f:
        json.dump(final_results, output_f, indent=4)

    # Stop Spark
    sc.stop()





if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    task1(input_file, output_file)
