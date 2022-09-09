from .dataset_ABSC import Tweet_DataSet, Restaurant_16_DataSet, Restaurant_14_DataSet, Restaurant_15_DataSet, Lap_14_DataSet
from .dataset_PSC import IMDB, YELP13, YELP14

DATASET_MAP = {
    'twt': Tweet_DataSet,
    'rst_16': Restaurant_16_DataSet,
    'rst_15': Restaurant_15_DataSet,
    'rst_14': Restaurant_14_DataSet,
    'lpt_14': Lap_14_DataSet,
}

PSC_DATASET_MAP = {
    'imdb': IMDB,
    'yelp_13': YELP13,
    'yelp_14': YELP14
}