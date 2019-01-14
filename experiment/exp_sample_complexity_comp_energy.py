import sys;
import os

sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation, prepare_data
import pandas as pd
from subsampling import Subsampling
from user_feedback import Similarity
from scipy.misc import comb
from deep_metric_learning import Deep_Metric
from deep_metric_learning_duplicate import Deep_Metric_Duplicate
import numpy as np
import pickle


print('experiement sample complexity script')
util = Utilities()
pe = PerformanceEvaluation()

# load dataset
day_profile_all = pd.read_pickle('dataset/dataframe_all_energy.pkl')
day_profile_all = day_profile_all.fillna(0)
res = 4

# define use case
interest = 'window-usage'
window = [17, 21]
rep_mode = 'mean'

# specify the data set for learning and for sanitization
n_rows = 50
pub_size = 80
day_profile = day_profile_all.iloc[:n_rows,0::res]
day_profile_learning = day_profile_all.iloc[n_rows:n_rows+pub_size,0::res]

anonymity_level = 4

sanitized_profile_best = util.sanitize_data(day_profile, distance_metric='self-defined',
                                            anonymity_level=anonymity_level, rep_mode=rep_mode,
                                            mode=interest, window=window)
sanitized_profile_baseline = util.sanitize_data(day_profile, distance_metric='euclidean',
                                                anonymity_level=anonymity_level, rep_mode=rep_mode)
loss_best_metric = pe.get_information_loss(data_gt=day_profile, data_sanitized=sanitized_profile_best,
                                           window=window)
loss_generic_metric = pe.get_information_loss(data_gt=day_profile,
                                              data_sanitized=sanitized_profile_baseline.round(),
                                              window=window)

df_subsampled_from = day_profile_learning.drop_duplicates()
subsample_size_max = int(comb(len(df_subsampled_from), 2))
print('total number of pairs is %s' % subsample_size_max)

sp = Subsampling(data=df_subsampled_from)
data_pair_all, data_pair_all_index = sp.uniform_sampling(subsample_size=subsample_size_max, seed=0)

sim = Similarity(data=data_pair_all)
sim.extract_interested_attribute(interest='statistics', stat_type=interest, window=window)
similarity_label_all, class_label_all = sim.label_via_silhouette_analysis(range_n_clusters=range(2, 8))
similarity_label_all_series = pd.Series(similarity_label_all)
similarity_label_all_series.index = data_pair_all_index
print('similarity balance is %s'% [sum(similarity_label_all),len(similarity_label_all)])




sample_size_vec = np.concatenate(([0.01,0.05],np.arange(0.1,1.1,0.1))) #np.array([1e-3,5e-3,1e-2,5e-2,1e-1,2e-2,3e-1])#
mc_num = 5
seed_vec = np.arange(mc_num)
loss_unif_mc_linear = np.zeros((mc_num,len(sample_size_vec)))
for mc_i in range(len(seed_vec)):
    for j in range(len(sample_size_vec)):
        sample_size = sample_size_vec[j]*subsample_size_max
        pairdata, pairdata_idx = sp.uniform_sampling(subsample_size=int(sample_size), seed=seed_vec[mc_i])
        pairdata_label = similarity_label_all_series.loc[pairdata_idx]

        x1_train, x2_train, y_train, x1_test, x2_test, y_test = prepare_data(pairdata, pairdata_label, 1)
        # dm = Deep_Metric(mode='relu')
        # dm.train(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
        lm = Deep_Metric_Duplicate(mode='linear')
        lm.train(x1_train, x2_train, y_train, x1_test, x2_test, y_test)

        sanitized_profile_linear = util.sanitize_data(day_profile, distance_metric="deep",
                                                      anonymity_level=anonymity_level,
                                                      rep_mode=rep_mode, deep_model=lm)
        #
        # sanitized_profile_deep = util.sanitize_data(day_profile, distance_metric="deep",
        #                                             anonymity_level=anonymity_level,
        #                                             rep_mode=rep_mode, deep_model=dm)

        loss_learned_metric_linear = pe.get_information_loss(data_gt=day_profile,
                                                             data_sanitized=sanitized_profile_linear.round(),
                                                             window=window)
        #
        # loss_learned_metric_deep = pe.get_information_loss(data_gt=day_profile,
        #                                                    data_sanitized=sanitized_profile_deep.round(),
        #                                                    window=window)

        loss_unif_mc_linear[mc_i,j] = loss_learned_metric_linear

        print('====================')
        print('random state %s ' % mc_i)
        print('sample size %s '% sample_size)
        print("information loss with best metric %s" % loss_best_metric)
        print("information loss with generic metric %s" % loss_generic_metric)
        print("information loss with linear metric deep  %s" % loss_learned_metric_linear)


    with open('result_linear_new/comp_unif_energy.pickle', 'wb') as f:
        pickle.dump([loss_best_metric,loss_generic_metric, loss_unif_mc_linear,subsample_size_max], f)






