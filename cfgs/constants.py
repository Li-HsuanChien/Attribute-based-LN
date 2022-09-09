import os

VECTOR_NAME = 'glove.840B.300d'

DATASET_PATH = 'corpus'
CKPTS_PATH = 'ckpts'
LOG_PATH = 'logs'
TASK_TYPE = {'ABSC', 'PSC'}

ABSC_DATASET_PATH_MAP = {
    "twt": os.path.join(DATASET_PATH, 'ABSC', 'tweet'),
    "lpt_14": os.path.join(DATASET_PATH, 'ABSC', 'semeval-14', 'laptop'),
    "rst_14": os.path.join(DATASET_PATH, 'ABSC', 'semeval-14'),
    "rst_15": os.path.join(DATASET_PATH, 'ABSC', 'semeval-15'),
    "rst_16": os.path.join(DATASET_PATH, 'ABSC', 'semeval-16'),
}

ABSC_DATASET_NUMCLASS = {
    "twt": 3,
    "lpt_14": 3,
    "rst_14": 3,
    "rst_15": 3,
    "rst_16": 3,
}

PSC_DATASET_PATH_MAP = {
    "imdb": os.path.join(DATASET_PATH, 'PSC', 'imdb'),
    "yelp_13":os.path.join(DATASET_PATH, 'PSC', 'yelp_13'),
    "yelp_14":os.path.join(DATASET_PATH, 'PSC', 'yelp_14')
}


PSC_DATASET_NUMCLASS = {
    "imdb": 10,
    "yelp_13": 5,
    "yelp_14": 5,
}


for directory in [CKPTS_PATH, LOG_PATH]:
    if not os.path.exists(directory): os.makedirs(directory)