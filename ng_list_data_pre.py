import pickle
import numpy as np

encoding = 'iso-8859-1'

def _random_sampling(item_tuple, num_sample):
    '''
    poi_type_tuple: (Type1, Type2,...TypeM)
    '''

    type_list = list(item_tuple)
    if len(type_list) > num_sample:
        return tuple(np.random.choice(type_list, num_sample, replace=False))
    elif len(type_list) == num_sample:
        return item_tuple
    else:
        return tuple(np.random.choice(type_list, num_sample, replace=True))

class NeighborGraph():
    def __init__(self, neighbor_tuple, neg_samples): 
        '''
        Note we only use id to indicate points
        A neighbor_tuple: [CenterPT, [ContextPT1, ContextPT2, ..., ContextPTN], training/validation/test]
        A full tuple in data sample: (neighbor_tuple, NegativeSampleNum, (NegPT1, NegPT2, ...,NegPTJ))
        '''
        self.center_pt = neighbor_tuple[0]
        self.context_pts = tuple(neighbor_tuple[1]) # a tuple of context points id
        self.data_mode = neighbor_tuple[2] # training/validation/test
        if neg_samples is not None:
            self.neg_samples = neg_samples # a tuple of negative samples id

        self.sample_context_pts = None
        self.sample_neg_pts = None

    def sample_neighbor(self, num_sample):
        self.sample_context_pts = _random_sampling(self.context_pts, num_sample)

    def sample_neg(self, num_neg_sample):
        self.sample_neg_pts = list(_random_sampling(self.neg_samples, num_neg_sample))

    def __hash__(self):
         return hash((self.center_pt, self.context_pts, self.data_mode, self.neg_samples))

    def __eq__(self, other):
        '''
        The euqavalence between two neighborgraph
        '''
        return (self.center_pt, self.context_pts, self.data_mode, self.neg_samples) == (other.center_pt, other.context_pts, other.data_mode, other.neg_samples)

    def __neq__(self, other):
        return self.__hash__() != other.__hash__()

    def serialize(self):
        '''
        Serialize the current NeighborGraph() object as an entry for train/val/test data samples
        '''
        return (self.center_pt, self.context_pts, self.data_mode, self.neg_samples)

    @staticmethod
    def deserialize(serial_info):
        '''
        Given a entry (serial_info) in train/val/test data samples
        parse it as NeighborGraph() object
        '''
        return NeighborGraph(serial_info[:3], serial_info[3])

        
def load_ng(data_file):
    raw_info = pickle.load(open(data_file, "rb"),encoding=encoding)
    return [NeighborGraph.deserialize(info) for info in raw_info]