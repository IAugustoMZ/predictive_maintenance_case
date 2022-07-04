# imports
import os
import sys
import warnings
import pandas as pd
import datetime

# append the MLutils module
sys.path.append(os.getcwd())
from modules.utils.ml_utils import MLutils

# ignore warnings
warnings.filterwarnings('ignore')

# define constants
DATASOURCE = os.path.join(os.getcwd(), 'data/3_model_data/')
DATANAME = 'rlu_model_data.csv'
target = 'rlu'

# instance of class
mlu = MLutils()

# load data
print(f'{datetime.datetime.now()} - Loding data')
data = pd.read_csv(os.path.join(DATASOURCE, DATANAME))

print(f'{datetime.datetime.now()} - Making feature selection')

# make feature selection
mlu.select_features(
    data=data,
    target=target,
    n_folds=3
)

# create experiment name
experiment_name = f'model_{target}'

for alias in ['rf', 'ext', 'gb', 'ada', 'xgb']:

    print(f'{datetime.datetime.now()} - Starting model training - {alias}')
    # create model pipeline
    mlu.create_model_pipeline(model_alias=alias)

    # train model
    mlu.model_fitting(experiment_name)
