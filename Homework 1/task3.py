import sys
from pyspark import SparkConf, SparkContext
import json
import time


def task3(review_filepath, business_filepath, output_file_a, output_file_b):
    # Spark configuration
    conf = SparkConf().setAppName("task3")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # Review and business files to get the stars and city
    reviews = sc.textFile(review_filepath)
    cleaned_reviews = reviews.map(lambda x: json.loads(x))
    stars = cleaned_reviews.map(lambda x: (x["business_id"], x["stars"]))


    business = sc.textFile(business_filepath)
    cleaned_business = business.map(lambda x: json.loads(x))
    city = cleaned_business.map(lambda x: (x["business_id"], x["city"]))

    # Joined RDD for city and stars
    city_stars = stars.join(city).map(lambda x: (x[1][1], (x[1][0], 1)))
    total_city_stars = city_stars.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    city_average = total_city_stars.mapValues(lambda avg: avg[0] / avg[1])
    sorted_avg = city_average.sortBy(lambda x: (-x[1], x[0]))

    # Write data into file
    with open(output_file_a, 'w') as output_a:
        output_a.write("city,stars\n")
        for city, avg_stars in sorted_avg.collect():
            output_a.write(f"{city}, {avg_stars}\n")

    # Calculate M1 execution time = loading time + time to create and collect averages (PYTHON)
    m1_start_time = time.time()
    m1_city_avg = city_average.collect()
    m1_sorted = sorted(m1_city_avg, key=lambda x: (-x[1], x[0]))[:10]
    m1_final_time = time.time() - m1_start_time

    # Calculate M2 execution time = loading time + time to create and collect averages (SPARK)
    m2_start_time = time.time()
    m2_city = city_average.take(10)
    m2_final_time = time.time() - m2_start_time

    # Results for M1 and M2 execution time & reasoning
    results = {"m1": m1_final_time, "m2": m2_final_time,
        "reason": "M1 ran faster which was when I used Python. However M2 should run faster because we have a large dataset "
                  "that we are processing and loading. This means that Python would take much longer since it doesn't have the"
                  "capabilities that Spark does. Python is runs slower to be able to compute larger datasets whereas Spark is made"
                  "for big data meaning it should run faster in this scenario. However I believe that it must have cpatured a small part of"
                "the data to run on which is why M1 is faster."}

    # Write the execution time results into the file
    with open(output_file_b, 'w') as output_b:
        json.dump(results, output_b, indent=4, separators=(',', ':'))

    # Stop Spark
    sc.stop()






if __name__ == '__main__':
    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_file_a = sys.argv[3]
    output_file_b = sys.argv[4]
    task3(review_filepath, business_filepath, output_file_a, output_file_b)