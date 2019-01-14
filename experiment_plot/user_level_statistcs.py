import sys; import os
sys.path.append(os.path.abspath("./"))
import pandas as pd
import numpy as np
from helper import Utilities, PerformanceEvaluation
import matplotlib.pyplot as plt
import pickle
from data_statistics import OccupancyStatistics
import scipy.stats

day_profile = pd.read_pickle('dataset/dataframe_all_user.pickle')
# res = 15
# day_profile = day_profile.iloc[:,0::res]
ncols = len(day_profile.columns)
rep_mode = 'mean'
util = Utilities()
anonymity_level_vec = np.arange(2,8)

with open('result_linear_new/user_level_profiles.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
   _,sanitized_profile_baseline_list = pickle.load(f)

fontsize = 18
legendsize = 12
bins =24

# 5-anonymization
i = 4
print('anonymization level %s'%anonymity_level_vec[i])
sanitized_profile = sanitized_profile_baseline_list[i]
sanitized_profile = sanitized_profile.round()

# Normalized histograms of arrival times
stat_gt = OccupancyStatistics(day_profile)
arrival_gt = stat_gt.get_arrival_time(flag=0)/12
arrival_gt = arrival_gt.dropna()
depart_gt = stat_gt.get_departure_time(flag=0)/12
depart_gt = depart_gt.dropna()
usage_gt = stat_gt.get_total_usage()/12
usage_gt = usage_gt.dropna()

stat_sn = OccupancyStatistics(sanitized_profile)
arrival_sn = stat_sn.get_arrival_time(flag=0)/12
arrival_sn = arrival_sn.dropna()
depart_sn = stat_sn.get_departure_time(flag=0)/12
depart_sn = depart_sn.dropna()
usage_sn = stat_sn.get_total_usage()/12
usage_sn = usage_sn.dropna()


bins = 24
# Normalized histograms of arrival times
plt.figure()
_,nbins,_ = plt.hist(arrival_gt,alpha=0.4,label='Original database',normed=True,bins=bins)
plt.hist(arrival_sn,alpha=0.4,label=str(anonymity_level_vec[i])+'-anonymized database',normed=True,bins=nbins)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of arrival times',fontsize=fontsize)
plt.ylim([0,0.9806])
plt.show()
ymin_arrival,ymax_arrival = plt.ylim()
xmin_arrival,xmax_arrival = plt.xlim()
plt.savefig('../pics/hist_arrival_user_'+str(anonymity_level_vec[i]))


# Normalized histograms of departure times
plt.figure()
_,nbins,_ = plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=bins)
plt.hist(depart_sn,alpha=0.4,label=str(anonymity_level_vec[i])+'-anonymized database',normed=True,bins=nbins)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of departure times',fontsize=fontsize)
plt.show()
ymin_depart,ymax_depart = plt.ylim()
xmin_depart,xmax_depart = plt.xlim()
plt.savefig('../pics/hist_departure_user_'+str(anonymity_level_vec[i]))


# Normalized histograms of total use time
plt.figure()
_,nbins,_ = plt.hist(usage_gt,alpha=0.4,label='Original database',normed=True,bins=bins)
plt.hist(usage_sn,alpha=0.4,label=str(anonymity_level_vec[i])+'-anonymized database',normed=True,bins=nbins)
plt.legend(fontsize=legendsize)
plt.xlabel('Hours',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of total usage',fontsize=fontsize)
plt.show()
ymin_tot,ymax_tot = plt.ylim()
xmin_tot,xmax_tot = plt.xlim()
plt.savefig('../pics/hist_tot_user_'+str(anonymity_level_vec[i]))






# 2-anonymization
i = 0
print('anonymization level %s'%anonymity_level_vec[i])
sanitized_profile = sanitized_profile_baseline_list[i]
sanitized_profile = sanitized_profile.round()

stat_gt = OccupancyStatistics(day_profile)
arrival_gt = stat_gt.get_arrival_time(flag=0)/12
arrival_gt = arrival_gt.dropna()
depart_gt = stat_gt.get_departure_time(flag=0)/12
depart_gt = depart_gt.dropna()
usage_gt = stat_gt.get_total_usage()/12
usage_gt = usage_gt.dropna()

stat_sn = OccupancyStatistics(sanitized_profile)
arrival_sn = stat_sn.get_arrival_time(flag=0)/12
arrival_sn = arrival_sn.dropna()
depart_sn = stat_sn.get_departure_time(flag=0)/12
depart_sn = depart_sn.dropna()
usage_sn = stat_sn.get_total_usage()/12
usage_sn = usage_sn.dropna()



# Normalized histograms of arrival times
plt.figure()
_,nbins,_ = plt.hist(arrival_gt,alpha=0.4,label='Original database',normed=True,bins=bins )
plt.hist(arrival_sn,alpha=0.4,label=str(anonymity_level_vec[i])+'-anonymized database',normed=True,bins=nbins)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of arrival times',fontsize=fontsize)
plt.xlim((xmin_arrival,xmax_arrival))
plt.ylim((ymin_arrival,ymax_arrival+0.2))
plt.show()
plt.savefig('../pics/hist_arrival_user_'+str(anonymity_level_vec[i]))


# Normalized histograms of departure times
plt.figure()
_,nbins,_ = plt.hist(depart_gt,alpha=0.4,label='Original database',normed=True,bins=bins )
plt.hist(depart_sn,alpha=0.4,label=str(anonymity_level_vec[i])+'-anonymized database',normed=True,bins=nbins)
plt.legend(fontsize=legendsize)
plt.xlabel('Hour index',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of departure times',fontsize=fontsize)
plt.xlim((xmin_depart,xmax_depart ))
plt.ylim((ymin_depart,ymax_depart ))
plt.show()
plt.savefig('../pics/hist_departure_user_'+str(anonymity_level_vec[i]))


# Normalized histograms of total use time
plt.figure()
_,nbins,_ = plt.hist(usage_gt,alpha=0.4,label='Original database',normed=True,bins=bins)
plt.hist(usage_sn,alpha=0.4,label=str(anonymity_level_vec[i])+'-anonymized database',normed=True,bins=nbins)
plt.legend(fontsize=legendsize)
plt.xlabel('Hours',fontsize=fontsize)
plt.ylabel('Frequency',fontsize=fontsize)
plt.title('Normalized histogram of total usage',fontsize=fontsize)
plt.xlim((xmin_tot,xmax_tot))
plt.ylim((ymin_tot,ymax_tot))
plt.show()
plt.savefig('../pics/hist_tot_user_'+str(anonymity_level_vec[i]))



