import pickle
import matplotlib.pyplot as plt
import numpy as np
from helper import Utilities, PerformanceEvaluation,prepare_data,get_ground_truth_distance
import pandas as pd
from user_feedback import Similarity
from scipy.misc import comb
import pickle
from subsampling import Subsampling
import pdb
import scipy.stats

## lunch time
fontsize = 18
with open('result_linear_new/lunchtime.pickle', 'rb') as f:
    anonymity_levels, losses_best, losses_generic, losses_linear, \
    losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)


with open('result_linear_new/lunchtime_euclidean_dist.pickle', 'rb') as f:
   distances_eu = pickle.load(f)


# print correlation coefficient
dist_gt = np.array([])
dist_dm = np.array([])
dist_lm = np.array([])
dist_eu = np.array([])
mc_num = 5
ai = 2
corr_eu= np.empty(mc_num)
corr_lm = np.empty(mc_num)
corr_dm = np.empty(mc_num)
for mc_i in range(mc_num):
    dist_gt_temp =distances_gt[(ai,mc_i)]
    dist_gt = np.concatenate((dist_gt,dist_gt_temp))
    dist_lm_temp = np.squeeze(distances_lm[(ai, mc_i)].reshape((1,len(distances_lm[(ai, mc_i)]))))
    dist_lm= np.concatenate((dist_lm,dist_lm_temp))
    dist_dm_temp = np.squeeze(distances_dm[(ai, mc_i)].reshape((1, len(distances_dm[(ai, mc_i)]))))
    dist_dm = np.concatenate((dist_dm,dist_dm_temp))
    dist_eu_temp = np.squeeze(distances_eu[(ai, mc_i)].reshape((1, len(distances_eu[(ai, mc_i)]))))
    dist_eu = np.concatenate((dist_eu,dist_eu_temp))

    corr_eu[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_eu_temp)[0]
    corr_lm[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_lm_temp)[0]
    corr_dm[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_dm_temp)[0]
print('eu is (%s)' % corr_eu)
print('lm is (%s)' % corr_lm)
print('dm is (%s)' % corr_dm)

print('eu mean is (%s)' % np.mean(corr_eu))
print('lm mean is (%s)' % np.mean(corr_lm))
print('dm mean is (%s)' % np.mean(corr_dm))



## peaktime
fontsize = 18
with open('result_linear_new/peaktime.pickle', 'rb') as f:
    anonymity_levels, losses_best, losses_generic, losses_linear, \
    losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)


with open('result_linear_new/peaktime_euclidean_dist.pickle', 'rb') as f:
   distances_eu = pickle.load(f)


# print correlation coefficient
dist_gt = np.array([])
dist_dm = np.array([])
dist_lm = np.array([])
dist_eu = np.array([])
mc_num = 5
ai = 2
corr_eu= np.empty(mc_num)
corr_lm = np.empty(mc_num)
corr_dm = np.empty(mc_num)
for mc_i in range(mc_num):
    dist_gt_temp =distances_gt[(ai,mc_i)]
    dist_gt = np.concatenate((dist_gt,dist_gt_temp))
    dist_lm_temp = np.squeeze(distances_lm[(ai, mc_i)].reshape((1,len(distances_lm[(ai, mc_i)]))))
    dist_lm= np.concatenate((dist_lm,dist_lm_temp))
    dist_dm_temp = np.squeeze(distances_dm[(ai, mc_i)].reshape((1, len(distances_dm[(ai, mc_i)]))))
    dist_dm = np.concatenate((dist_dm,dist_dm_temp))
    dist_eu_temp = np.squeeze(distances_eu[(ai, mc_i)].reshape((1, len(distances_eu[(ai, mc_i)]))))
    dist_eu = np.concatenate((dist_eu,dist_eu_temp))

    corr_eu[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_eu_temp)[0]
    corr_lm[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_lm_temp)[0]
    corr_dm[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_dm_temp)[0]
print('eu is (%s)' % corr_eu)
print('lm is (%s)' % corr_lm)
print('dm is (%s)' % corr_dm)

print('eu mean is (%s)' % np.mean(corr_eu))
print('lm mean is (%s)' % np.mean(corr_lm))
print('dm mean is (%s)' % np.mean(corr_dm))




## arrival time
fontsize = 18
with open('result_nonlinear/arrivaltime.pickle', 'rb') as f:
    anonymity_levels, losses_best, losses_generic, losses_linear, \
    losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)


with open('result_nonlinear/arrivaltime_euclidean_dist.pickle', 'rb') as f:
   distances_eu = pickle.load(f)


# print correlation coefficient
dist_gt = np.array([])
dist_dm = np.array([])
dist_lm = np.array([])
dist_eu = np.array([])
mc_num = 5
ai = 2
corr_eu= np.empty(mc_num)
corr_lm = np.empty(mc_num)
corr_dm = np.empty(mc_num)
for mc_i in range(mc_num):
    dist_gt_temp =distances_gt[(ai,mc_i)]
    dist_gt = np.concatenate((dist_gt,dist_gt_temp))
    dist_lm_temp = np.squeeze(distances_lm[(ai, mc_i)].reshape((1,len(distances_lm[(ai, mc_i)]))))
    dist_lm= np.concatenate((dist_lm,dist_lm_temp))
    dist_dm_temp = np.squeeze(distances_dm[(ai, mc_i)].reshape((1, len(distances_dm[(ai, mc_i)]))))
    dist_dm = np.concatenate((dist_dm,dist_dm_temp))
    dist_eu_temp = np.squeeze(distances_eu[(ai, mc_i)].reshape((1, len(distances_eu[(ai, mc_i)]))))
    dist_eu = np.concatenate((dist_eu,dist_eu_temp))

    corr_eu[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_eu_temp)[0]
    corr_lm[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_lm_temp)[0]
    corr_dm[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_dm_temp)[0]
print('eu is (%s)' % corr_eu)
print('lm is (%s)' % corr_lm)
print('dm is (%s)' % corr_dm)

print('eu mean is (%s)' % np.mean(corr_eu))
print('lm mean is (%s)' % np.mean(corr_lm))
print('dm mean is (%s)' % np.mean(corr_dm))

# learned distance
dist_gt = np.array([])
dist_dm = np.array([])
dist_lm = np.array([])
dist_eu = np.array([])
mc_num = 5
for ai in [2]:
    for mc_i in range(mc_num):
        dist_gt_temp =distances_gt[(ai,mc_i)]
        dist_gt = np.concatenate((dist_gt,dist_gt_temp))
        dist_lm_temp = np.squeeze(distances_lm[(ai, mc_i)].reshape((1,len(distances_lm[(ai, mc_i)]))))
        dist_lm= np.concatenate((dist_lm,dist_lm_temp))
        dist_dm_temp = np.squeeze(distances_dm[(ai, mc_i)].reshape((1, len(distances_dm[(ai, mc_i)]))))
        dist_dm = np.concatenate((dist_dm,dist_dm_temp))
        dist_eu_temp = np.squeeze(distances_eu[(ai, mc_i)].reshape((1, len(distances_eu[(ai, mc_i)]))))
        dist_eu = np.concatenate((dist_eu,dist_eu_temp))



fit_lm = np.polyfit(dist_gt,dist_lm,5)
fit_dm = np.polyfit(dist_gt,dist_dm,5)
fit_lm_fn = np.poly1d(fit_lm)
fit_dm_fn = np.poly1d(fit_dm)
plt.figure()
# plt.plot(np.sort(dist_gt),fit_lm_fn(np.sort(dist_gt)),'g--',label='Fitted trend line for Linear metric')
# plt.plot(np.sort(dist_gt),fit_dm_fn(np.sort(dist_gt)),'r--',label='Fitted trend line for Nonlinear metric')
plt.plot(dist_lm,dist_gt,'go',alpha=0.1,label='Linear metric')
plt.plot(dist_dm,dist_gt,'ro',alpha=0.1,label='Nonlinear metric')
plt.plot(dist_eu,dist_gt,'bo',alpha=0.1,label='Euclidean metric')
plt.xlabel('Ground truth distance',fontsize=fontsize)
plt.ylabel('Learned distance',fontsize=fontsize)
plt.title('Arrival time example', fontsize=18)
plt.legend()
plt.show()

# plt.figure()
# plt.plot(np.sort(dist_gt),fit_lm_fn(np.sort(dist_gt)),'g--',label='Fitted trend line for Linear metric')
# plt.plot(np.sort(dist_gt),fit_dm_fn(np.sort(dist_gt)),'r--',label='Fitted trend line for Nonlinear metric')

# plt.plot(dist_gt,dist_dm,'ro',alpha=0.1,label='Nonlinear metric')
plt.xlabel('Ground truth distance',fontsize=fontsize)
plt.ylabel('Learned distance',fontsize=fontsize)
plt.title('Arrival time example', fontsize=18)
plt.legend()
plt.show()



############# departure time  #############
fontsize = 18
with open('result_nonlinear/departtime.pickle', 'rb') as f:
    anonymity_levels, losses_best, losses_generic, losses_linear, \
    losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)


with open('result_nonlinear/departtime_euclidean_dist.pickle', 'rb') as f:
   distances_eu = pickle.load(f)

# print correlation coefficient
dist_gt = np.array([])
dist_dm = np.array([])
dist_lm = np.array([])
dist_eu = np.array([])
mc_num = 5
ai = 2
corr_eu= np.empty(mc_num)
corr_lm = np.empty(mc_num)
corr_dm = np.empty(mc_num)
for mc_i in range(mc_num):
    dist_gt_temp =distances_gt[(ai,mc_i)]
    dist_gt = np.concatenate((dist_gt,dist_gt_temp))
    dist_lm_temp = np.squeeze(distances_lm[(ai, mc_i)].reshape((1,len(distances_lm[(ai, mc_i)]))))
    dist_lm= np.concatenate((dist_lm,dist_lm_temp))
    dist_dm_temp = np.squeeze(distances_dm[(ai, mc_i)].reshape((1, len(distances_dm[(ai, mc_i)]))))
    dist_dm = np.concatenate((dist_dm,dist_dm_temp))
    dist_eu_temp = np.squeeze(distances_eu[(ai, mc_i)].reshape((1, len(distances_eu[(ai, mc_i)]))))
    dist_eu = np.concatenate((dist_eu,dist_eu_temp))

    corr_eu[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_eu_temp)[0]
    corr_lm[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_lm_temp)[0]
    corr_dm[mc_i] = scipy.stats.pearsonr(dist_gt_temp, dist_dm_temp)[0]
print('eu is (%s)' % corr_eu)
print('lm is (%s)' % corr_lm)
print('dm is (%s)' % corr_dm)

print('eu mean is (%s)' % np.mean(corr_eu))
print('lm mean is (%s)' % np.mean(corr_lm))
print('dm mean is (%s)' % np.mean(corr_dm))

# learned distance
dist_gt = np.array([])
dist_dm = np.array([])
dist_lm = np.array([])
dist_eu = np.array([])
mc_num = 1
for ai in [3]:
    for mc_i in [2]:
        dist_gt = np.concatenate((dist_gt,distances_gt[(ai,mc_i)]))
        dist_lm= np.concatenate((dist_lm,
                                 np.squeeze(
                                     distances_lm[(ai, mc_i)].reshape((1,len(distances_lm[(ai, mc_i)]))))))
        dist_dm = np.concatenate((dist_dm,
                                  np.squeeze(
                                      distances_dm[(ai, mc_i)].reshape((1, len(distances_dm[(ai, mc_i)]))))))
        dist_eu = np.concatenate((dist_eu,
                                  np.squeeze(
                                      distances_eu[(ai, mc_i)].reshape((1, len(distances_eu[(ai, mc_i)]))))))


fit_lm = np.polyfit(dist_gt,dist_lm,5)
fit_dm = np.polyfit(dist_gt,dist_dm,5)
fit_lm_fn = np.poly1d(fit_lm)
fit_dm_fn = np.poly1d(fit_dm)
plt.figure()
# plt.plot(np.sort(dist_gt),fit_lm_fn(np.sort(dist_gt)),'g--',label='Fitted trend line for Linear metric')
# plt.plot(np.sort(dist_gt),fit_dm_fn(np.sort(dist_gt)),'r--',label='Fitted trend line for Nonlinear metric')
plt.plot(dist_lm,dist_gt,'go',alpha=0.1,label='Linear metric')
plt.plot(dist_dm,dist_gt,'ro',alpha=0.1,label='Nonlinear metric')
plt.plot(dist_eu,dist_gt,'bo',alpha=0.1,label='Euclidean metric')
plt.xlabel('Ground truth distance',fontsize=fontsize)
plt.ylabel('Learned distance',fontsize=fontsize)
plt.title('Arrival time example', fontsize=18)
plt.legend()
plt.show()

# plt.figure()
# plt.plot(np.sort(dist_gt),fit_lm_fn(np.sort(dist_gt)),'g--',label='Fitted trend line for Linear metric')
# plt.plot(np.sort(dist_gt),fit_dm_fn(np.sort(dist_gt)),'r--',label='Fitted trend line for Nonlinear metric')

# plt.plot(dist_gt,dist_dm,'ro',alpha=0.1,label='Nonlinear metric')
plt.xlabel('Ground truth distance',fontsize=fontsize)
plt.ylabel('Learned distance',fontsize=fontsize)
plt.title('Arrival time example', fontsize=18)
plt.legend()
plt.show()

