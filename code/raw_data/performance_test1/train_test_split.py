import sys
import numpy as np


def main():
    input_filename = sys.argv[1]
    train_prop = float(sys.argv[2])

    idx = 0
    r = np.random.uniform(0, 1, 10000)

    with open(input_filename, 'r') as input_fp, \
            open(input_filename + '.train', 'w') as train_fp, \
            open(input_filename + '.test', 'w') as test_fp:
        for line in input_fp:
            if r[idx] < train_prop:
                print >>train_fp, line.strip()
            else:
                print >>test_fp, line.strip()

            idx += 1
            if idx >= 10000:
                idx = 0
                r = np.random.uniform(0, 1, 10000)


if __name__ == '__main__': main()
