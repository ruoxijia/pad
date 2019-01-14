import sys; import os
sys.path.append(os.path.abspath("./"))
from helper import Utilities, PerformanceEvaluation,prepare_data,get_ground_truth_distance
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import pickle
from scipy.misc import comb
import time
from subsampling import Subsampling
from user_feedback import Similarity
from deep_metric_learning_duplicate import Deep_Metric_Duplicate

day_profile = pd.read_pickle('dataset/dataframe_all_binary.pkl')
day_profile.index = range(len(day_profile.index))
day_profile = day_profile.iloc[0::5,:]
print('row num%s' %len(day_profile.index))
ncols = len(day_profile.columns)
nrows = len(day_profile.index)
npairs = int(comb(nrows,2))
interest = 'segment'
rep_mode = 'mean'
window = [5,20]

anonymity_level_vec = np.arange(2,8)
sample_vec = np.arange(100,npairs,100)#np.concatenate((np.array([1]),np.arange(2,21,4)))
d_vec = [24, 48, 96, 360]#np.arange(20,ncols,200)
comp_time = np.empty((len(d_vec),len(sample_vec)))


for di in range(len(d_vec)):
    sp = Subsampling(data=day_profile.iloc[:,0:d_vec[di]])
    print('sample dimension %s' % d_vec[di])


    for ni in range(len(sample_vec)):
        data_pair, data_pair_all_index = sp.uniform_sampling(subsample_size=sample_vec[ni], seed=0)
        print('sample number %s' % sample_vec[ni])
        sim = Similarity(data=data_pair)
        sim.extract_interested_attribute(interest=interest, window=window)
        similarity_label, data_subsample = sim.label_via_silhouette_analysis(range_n_clusters=range(2, 8))
        t0 = time.time()

        x1_train, x2_train, y_train, x1_test, x2_test, y_test = prepare_data(data_pair, similarity_label, 1)
        lm = Deep_Metric_Duplicate(mode='linear')
        lm.train(x1_train, x2_train, y_train, x1_test, x2_test, y_test)

        t1 = time.time()
        t = t1 - t0
        comp_time[di,ni] = t
        print('time elapsed %s' % t)

        with open('./result_linear_new/computation_issue.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([comp_time], f)





## visualization of computation time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm
import pdb
import pandas as pd
from scipy.misc import comb
with open('./result_linear_new/computation_issue.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    comp_time = pickle.load(f)
comp_time = comp_time[0]

day_profile = pd.read_pickle('dataset/dataframe_all_binary.pkl')
day_profile.index = range(len(day_profile.index))
day_profile = day_profile.iloc[0::5,:]
print('row num%s' %len(day_profile.index))
ncols = len(day_profile.columns)
nrows = len(day_profile.index)
npairs = int(comb(nrows,2))

# comp_time = comp_time[0:4,:]
sample_vec = np.arange(100,npairs,100)#np.concatenate((np.array([1]),np.arange(2,21,4)))
d_vec = [24, 48, 96, 360]#np.arange(20,ncols,200)

X,Y = np.meshgrid(d_vec,sample_vec)
fontsize = 18
fig = plt.figure()
ax = fig.gca(projection='3d')
color = 'r'
ax.plot_surface(X,Y,comp_time.transpose(),linewidth=0, antialiased=False,
                   alpha=0.3,color = color)
f = mpl.lines.Line2D([0], [0], linestyle="none", color=color, marker='o', alpha=0.3)
ax.set_xlabel('Row dimension',fontsize=fontsize)
ax.set_ylabel('Number of labeled pairs',fontsize=fontsize)
ax.set_zlabel('Computational time (s)',fontsize=fontsize)
# angle = 60
# ax.view_init(30, 150)
plt.show()



