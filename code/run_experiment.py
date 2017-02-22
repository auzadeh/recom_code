import sys
import pandas as pd
from itemitem_rec import *

COLD_START_FILTER = 1

def main():
    train_df = pd.read_csv(sys.argv[1], header=None, sep='\t', names=
            ['uid', 'bid', 'rating', 'tstamp'])
    recommender = W2VItemItemRecommender(sys.argv[2])
    recommender.train(train_df, 'uid', 'bid', 'rating')

    test_df = pd.read_csv(sys.argv[3], header=None, sep='\t', names=
            ['uid', 'bid', 'rating', 'tstamp'])

    user_not_found_counter = 0
    cold_start_counter = 0
    with open(sys.argv[4], 'w') as f:
        for idx, row in test_df.iterrows():
            try:
                predicted = recommender.predict(row['uid'], row['bid'],
                        COLD_START_FILTER)
            except UserNotInRecordError as e:
                user_not_found_counter += 1
                continue
            except ColdStartUserError as e:
                cold_start_counter += 1
                continue
            err = predicted - row['rating']
            print >>f, '%s\t%s\t%.5f\t%.5f' % (row['uid'], row['bid'], err, abs(err))



if __name__ == '__main__': main()
