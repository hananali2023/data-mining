import sys
import random
import binascii
import csv
import time
from blackbox import BlackBox

num_hashes = 15
large_prime = 69991

def myhashs(user_id):
    user_integer = int(binascii.hexlify(user_id.encode('utf8')), 16)
    return [((a_coeff * user_integer + b_coeff) % large_prime) for a_coeff, b_coeff in zip(a_params, b_params)]

def binary_numbers(x):
    binary = bin(x)[2:]
    return len(binary) - len(binary.rstrip('0'))

def flajolet_martin(stream):
    max_zeros = [0] * num_hashes
    for user in stream:
        hash_values = myhashs(user)
        for i, hash_val in enumerate(hash_values):
            max_zeros[i] = max(max_zeros[i], binary_numbers(hash_val))
    estimates = [2 ** zeros for zeros in max_zeros]
    estimates.sort()
    if len(estimates) % 2 == 0:
        return (estimates[len(estimates) // 2 - 1] + estimates[len(estimates) // 2]) / 2
    else:
        return estimates[len(estimates) // 2]

def main():
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    output_file = sys.argv[4]

    bx = BlackBox()

    start_time = time.time()

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "Ground Truth", "Estimation"])
        total_actual = 0
        total_estimated = 0
        for iteration in range(num_iterations):
            stream = bx.ask(input_file, stream_size)
            actual_unique = len(set(stream))
            estimated_unique = flajolet_martin(stream)
            writer.writerow([iteration, actual_unique, estimated_unique])
            total_actual += actual_unique
            total_estimated += estimated_unique
        accuracy_ratio = total_estimated / total_actual
    end_time = time.time()
    total_time = end_time - start_time


if __name__ == "__main__":
    random.seed(553)
    a_params = [random.randint(1, large_prime - 1) for _ in range(num_hashes)]
    b_params = [random.randint(0, large_prime - 1) for _ in range(num_hashes)]

    main()

