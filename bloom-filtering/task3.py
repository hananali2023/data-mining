import sys
import csv
import random
from blackbox import BlackBox
import time

reservoir_size = 100

def reservoir_sampling(stream, reservoir, sequence):
    for user in stream:
        sequence += 1
        if sequence <= reservoir_size:
            reservoir.append(user)
        else:
            if random.random() < reservoir_size / sequence:
                new_index = random.randint(0, reservoir_size - 1)
                reservoir[new_index] = user
    return sequence

def main():
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    random.seed(553)
    bx = BlackBox()

    reservoir = []
    seq_num = 0

    start_time = time.time()

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seqnum", "0_id", "20_id", "40_id", "60_id", "80_id"])
        for i in range(num_of_asks):
            stream = bx.ask(input_file, stream_size)
            seq_num = reservoir_sampling(stream, reservoir, seq_num)
            writer.writerow([
                seq_num,
                reservoir[0],
                reservoir[20],
                reservoir[40],
                reservoir[60],
                reservoir[80]
            ])

        end_time = time.time()
        total = end_time - start_time

if __name__ == "__main__":
    main()
