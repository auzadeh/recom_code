import math
import numpy as np


class ColdStartUserError(Exception):
    pass


class MeanBaselineRecommender(object):
    '''
        Baseline Recommender
            two stage conservative estimation of target
            first estimate mean of user target
            then estimate item's intrinsic quality In-RAM operations
        Warning: this implementation does not scale with training input size
    '''
    def __init__(self):
        # no initialization needed
        pass

    def train(self, train_data, uid_colname, iid_colname, target):
        # find bounds
        self._max_target = train_data[target].max()
        self._min_target = train_data[target].min()

        # find user means
        self._user_mean_records = train_data.groupby(uid_colname).mean()[target].to_dict()

        # find item deviation from means
        item_umean_diff = np.array([
            r[target] - self._user_mean_records[r[uid_colname]]
            for i, r in train_data.iterrows()
        ])
        train_data['item_umean_diff'] = item_umean_diff

        # find mean item deviation (estimated item quality)
        self._item_mean_diff = train_data.groupby(iid_colname).mean()['item_umean_diff'].to_dict()

    def predict(self, user_id, item_id, cold_start_filter=0):
        # take out a dictionary of item reocrds for the required-user
        user_mean = self._user_mean_records.get(user_id, None)
        if user_mean is None:
            raise UserNotInRecordError()

        item_mean_diff = self._item_mean_diff.get(item_id, 0.0)

        # prediction is user baseline + item quality
        prediction = user_mean + item_mean_diff

        if prediction < self._min_target:
            return self._min_target
        if prediction > self._max_target:
            return self._max_target
        return prediction
