import math


class UserNotInRecordError(Exception): pass
class ColdStartUserError(Exception):
    pass



class W2VItemItemRecommender(object):
    '''
        Item-Item Recommender using Word2Vec similarities of items
        In-RAM operations

        Warning: this implementation does not scale with training input size
    '''
    def __init__(self, embedding_fname):
        self._w2v_model = self._load_embedding(embedding_fname)

    def _load_embedding(self, fname):
        from gensim.models import Word2Vec
        return Word2Vec.load(fname)

    def train(self, train_data, uid_colname, iid_colname, target):
        '''
            W2VItemItem does not require explicit training.
            Its training is implicitly done while training the W2V embedding
            Prediction is generated at runtime

            Here we only re-organize input training data for faster prediction

            train_data should be a Pandas dataframe of at least
                the required three columns: user ids, item ids and target
        '''
        organizer_lambda = lambda d: \
                d[[iid_colname, target]].set_index(iid_colname).\
                to_dict()[target]

        self._user_mean_records = train_data.groupby(uid_colname).mean()[target].to_dict()

        # {uid: {iid: target}}
        self._user_target_records = train_data.groupby(uid_colname).apply(organizer_lambda).to_dict()
        self._max_target = train_data[target].max()
        self._min_target = train_data[target].min()

    def predict(self, user_id, item_id, cold_start_filter=0):
        # take out a dictionary of item reocrds for the required-user
        user_records = self._user_target_records.get(user_id, None)
        user_mean = self._user_mean_records.get(user_id, None)

        if user_records is None or user_mean is None:
            raise UserNotInRecordError()
        if len(user_records) <= cold_start_filter:
            raise ColdStartUserError()

        prediction = 0.0
        weight_sum = 0.0
        similarity_triger = False
        for record_iid, record_target in user_records.items():
            # take exponential of similarity because similarity can be negative
            try:
                weight = math.exp(self._w2v_model.similarity(item_id, record_iid))
                similarity_triger = True
            except KeyError as e:
                continue

            prediction += (record_target - user_mean) * weight
            weight_sum += weight
        if not similarity_triger:
            raise ColdStartUserError()
        # re-normalize so that prediction is within range of target column
        prediction /= weight_sum
        prediction += user_mean

        if prediction < self._min_target:
            return self._min_target
        if prediction > self._max_target:
            return self._max_target
        return prediction
