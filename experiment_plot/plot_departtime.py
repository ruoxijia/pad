import pickle
import matplotlib.pyplot as plt
import numpy as np


fontsize = 18
## lunch time
with open('result_nonlinear/departtime.pickle', 'rb') as f:
    anonymity_levels, losses_best, losses_generic, losses_linear, \
    losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)


plt.plot(anonymity_levels, losses_best,'o-', label='Ground truth metric')
plt.plot(anonymity_levels, losses_generic, 's-',label='Generic metric')
plt.errorbar(anonymity_levels, np.mean(losses_linear, axis=1), np.std(losses_linear, axis=1),
             fmt='^--', capthick=2,label='Linear metric')
plt.errorbar(anonymity_levels, np.mean(losses_deep, axis=1), np.std(losses_deep, axis=1),
             fmt='X--',capthick=2,label='Nonlinear metric')
plt.xlabel('Anonymity level', fontsize=18)
plt.ylabel('Information loss', fontsize=18)
plt.title('Departure time example', fontsize=18)
plt.legend()
plt.show()

# learned distance
dist_gt = np.array([])
dist_dm = np.array([])
dist_lm = np.array([])
mc_num = 1
for ai in range(len(anonymity_levels)):
    for mc_i in range(mc_num):
        dist_gt = np.concatenate((dist_gt,distances_gt[(ai,mc_i)]))
        dist_lm= np.concatenate((dist_lm,
                                 np.squeeze(
                                     distances_lm[(ai, mc_i)].reshape((1,len(distances_lm[(ai, mc_i)]))))))
        dist_dm = np.concatenate((dist_dm,
                                  np.squeeze(
                                      distances_dm[(ai, mc_i)].reshape((1, len(distances_dm[(ai, mc_i)]))))))


fit_lm = np.polyfit(dist_gt,dist_lm,1)
fit_dm = np.polyfit(dist_gt,dist_dm,1)
fit_lm_fn = np.poly1d(fit_lm)
fit_dm_fn = np.poly1d(fit_dm)
plt.figure()
plt.plot(np.sort(dist_gt),fit_lm_fn(np.sort(dist_gt)),'g--',label='Fitted trend line for Linear metric')
plt.plot(np.sort(dist_gt),fit_dm_fn(np.sort(dist_gt)),'r--',label='Fitted trend line for Nonlinear metric')
plt.plot(dist_gt,dist_lm,'go',alpha=0.1,label='Linear metric')
plt.plot(dist_gt,dist_dm,'ro',alpha=0.1,label='Nonlinear metric')
plt.xlabel('Ground truth distance',fontsize=fontsize)
plt.ylabel('Learned distance',fontsize=fontsize)
plt.title('Lunch time example', fontsize=18)
plt.legend()
plt.show()


# departure time (presanitized)
with open('result_nonlinear/departtime_presanitized.pickle', 'rb') as f:
    anonymity_levels, losses_best, losses_generic, losses_linear, \
    losses_deep, distances_lm, distances_dm, distances_gt = pickle.load(f)

plt.plot(anonymity_levels, losses_best,'o-', label='Ground truth metric')
plt.plot(anonymity_levels, losses_generic, 's-',label='Generic metric')
plt.errorbar(anonymity_levels, np.mean(losses_linear, axis=1), np.std(losses_linear, axis=1),
             fmt='^--', capthick=2,label='Linear metric')
plt.errorbar(anonymity_levels, np.mean(losses_deep, axis=1), np.std(losses_deep, axis=1),
             fmt='X--',capthick=2,label='Nonlinear metric')
plt.xlabel('Anonymity level', fontsize=18)
plt.ylabel('Information loss', fontsize=18)
plt.title('Departure time example', fontsize=18)
plt.legend()
plt.show()
