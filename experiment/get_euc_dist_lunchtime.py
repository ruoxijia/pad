import sys; import os
sys.path.append(os.path.abspath("./"))

import pickle
import matplotlib.pyplot as plt
import numpy as np
from helper import Utilities, PerformanceEvaluation ,prepare_data ,get_ground_truth_distance
import pandas as pd
from user_feedback import Similarity
from scipy.misc import comb
from subsampling import Subsampling
import pdb

# Initialization of some useful classes
util = Utilities()
pe = PerformanceEvaluation()

# load dataset
day_profile_all = pd.read_pickle('dataset/dataframe_all_binary.pkl')

day_profile_all = day_profile_all.fillna(0)
day_profile_all[day_profile_all > 0] = 1
res = 15

# define use case
interest = 'segment'
rep_mode = 'mean'
window = [11, 15]  # window specifies the starting and ending time of the period that the data user is interested in

# specify the data set for learning and for sanitization
n_rows = 80
pub_size = n_rows
day_profile = day_profile_all.iloc[:n_rows,0::res]
day_profile_learning = day_profile_all.iloc[n_rows:n_rows+pub_size,0::res]

# euclidean distance
mc_num = 5
frac = 0.8
anonymity_levels = np.arange(2,8)

distances_eu = {}

for i in range(len(anonymity_levels)):
    anonymity_level = anonymity_levels[i]
    for mc_i in range(mc_num):
        df_subsampled_from = day_profile_learning.sample(frac=frac, replace=False, random_state=mc_i)
        subsample_size_max = int(comb(len(df_subsampled_from), 2))
        print('total number of pairs is %s' % subsample_size_max)
        subsample_size = int(round(subsample_size_max))
        sp = Subsampling(data=df_subsampled_from)
        data_pair, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size, seed=0)

        sim = Similarity(data=data_pair)
        sim.extract_interested_attribute(interest=interest, window=window)
        similarity_label, data_subsample = sim.label_via_silhouette_analysis(range_n_clusters=range(2, 8))

        x1_train, x2_train, y_train, x1_test, x2_test, y_test = prepare_data(data_pair, similarity_label, 0.5)
        distance_eu = np.linalg.norm(x1_test-x2_test,ord=2,axis=1)

        distances_eu[(i, mc_i)] = distance_eu

        print('==========================')
        print('anonymity level index %s'% i)
        print('mc iteration %s' % mc_i)

with open('./result_linear_new/lunchtime_euclidean_dist.pickle', 'wb') as f:
    pickle.dump(distances_eu, f)