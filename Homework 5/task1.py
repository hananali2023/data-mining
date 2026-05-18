from blackbox import BlackBox
import binascii
import random
import csv
import sys
import time

bit_vector_size = 69997
prime_number = 78882
num_hash_functions = 3

def myhashs(user_id):
    user_integer = int(binascii.hexlify(user_id.encode('utf8')), 16)
    return [((coef_a * user_integer + coef_b) % prime_number) % bit_vector_size for coef_a, coef_b in zip(a_values, b_values)]

bit_vector = [0] * bit_vector_size
users = set()

def bloom_filter(user_id):
    hashes = myhashs(user_id)
    filtered = all(bit_vector[h] for h in hashes)
    for h in hashes:
        bit_vector[h] = 1
    return filtered

def FPR(stream, users):
    false_positives = 0
    total_negatives = 0
    for user in stream:
        if user not in users:
            total_negatives += 1
            if bloom_filter(user):
                false_positives += 1
        users.add(user)
    return false_positives / total_negatives if total_negatives else 0

def main():
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    blackbox = BlackBox()

    start_time = time.time()

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "FPR"])
        for i in range(num_of_asks):
            stream = blackbox.ask(input_file, stream_size)
            false_positive_rate = FPR(stream, users)
            writer.writerow([i, false_positive_rate])
    end_time = time.time()
    total = end_time - start_time

if __name__ == "__main__":
    random.seed(553)
    a_values = [random.randint(1, prime_number - 1) for _ in range(num_hash_functions)]
    b_values = [random.randint(0, prime_number - 1) for _ in range(num_hash_functions)]

    main()


