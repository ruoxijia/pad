import pandas as pd
from helper import Utilities

util = Utilities()

# step1: get original database to be published
day_profile = pd.read_pickle('../dataset/dataframe_all_binary.pkl')

# (optional) subsample the time series in each raw of the database
res = 15
day_profile = day_profile.iloc[0::5, 0::res]

# step2: specify the desired anonymity level
anonymity_level = 5

# util.sanitize_data will privatize the database according to the desired anonymity level
sanitized_profile = util.sanitize_data(day_profile, distance_metric='euclidean',
                                       anonymity_level=anonymity_level,rep_mode ='mean')


