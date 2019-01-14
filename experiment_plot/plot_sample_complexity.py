import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd


fontsize = 18
with open('result_linear_new/comp_unif_energy.pickle', 'rb') as f:
    loss_best_metric, loss_generic_metric, loss_unif_mc_linear, subsample_size_max = pickle.load(f)





mc_num = 5
num_con = 6
sample_size_vec= np.concatenate(([0.01,0.05],np.arange(0.1,1.1,0.1)))
sample_size_vec = sample_size_vec[:num_con]
eval_k = subsample_size_max * sample_size_vec
k_init = subsample_size_max * sample_size_vec[0]
run_sample_size = subsample_size_max * sample_size_vec[-1]

# loss_unif_all_deep = np.asarray(loss_unif_all_deep)
# loss_unif_all_deep_mean = np.mean(loss_unif_all_deep,axis=0)
# loss_unif_all_deep_std = np.std(loss_unif_all_deep,axis=0)

loss_unif_all_linear= np.asarray(loss_unif_mc_linear[:,:num_con])
loss_unif_all_linear_mean = np.mean(loss_unif_all_linear,axis=0)
loss_unif_all_linear_std = np.std(loss_unif_all_linear,axis=0)



plt.plot((k_init,run_sample_size ),(loss_best_metric,loss_best_metric),'--',label="Ground truth metric")
plt.plot((k_init,run_sample_size ),(loss_generic_metric,loss_generic_metric),'--',label="Generic metric")
plt.errorbar(eval_k,loss_unif_all_linear_mean,loss_unif_all_linear_std,fmt='^--',label='Learned metric')
plt.xlabel('Number of labeled data pairs', fontsize=18)
plt.ylabel('Information loss', fontsize=18)
plt.title('Sample efficiency', fontsize=18)
plt.legend()
plt.show()




### lunch time
with open('result_linear_new/comp_unif_lunchtime_new.pickle', 'rb') as f:
    loss_best_metric, loss_generic_metric, loss_unif_mc_linear, subsample_size_max = pickle.load(f)






mc_num = 5
num_con = 6
sample_size_vec= np.concatenate(([0.01,0.05],np.arange(0.1,1.1,0.1)))
sample_size_vec = sample_size_vec[:num_con]
eval_k = subsample_size_max * sample_size_vec
k_init = subsample_size_max * sample_size_vec[0]
run_sample_size = subsample_size_max * sample_size_vec[-1]

# loss_unif_all_deep = np.asarray(loss_unif_all_deep)
# loss_unif_all_deep_mean = np.mean(loss_unif_all_deep,axis=0)
# loss_unif_all_deep_std = np.std(loss_unif_all_deep,axis=0)

loss_unif_all_linear= np.asarray(loss_unif_mc_linear[:,:num_con])
loss_unif_all_linear_mean = np.mean(loss_unif_all_linear,axis=0)
loss_unif_all_linear_std = np.std(loss_unif_all_linear,axis=0)



plt.plot((k_init,run_sample_size ),(loss_best_metric,loss_best_metric),'--',label="Ground truth metric")
plt.plot((k_init,run_sample_size ),(loss_generic_metric,loss_generic_metric),'--',label="Generic metric")
plt.errorbar(eval_k,loss_unif_all_linear_mean,loss_unif_all_linear_std,fmt='^--',label='Learned metric')
plt.xlabel('Number of labeled data pairs', fontsize=18)
plt.ylabel('Information loss', fontsize=18)
plt.title('Sample efficiency', fontsize=18)
plt.legend()
plt.show()
