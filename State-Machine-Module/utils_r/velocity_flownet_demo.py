import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
gt = np.load('utils\\test_gt.npy',allow_pickle=True)
# 把gt 的 0 1 颠倒
# gt = 1 - gt
vel = np.load('utils\\final_scores_velocity.npy',allow_pickle=True)
optical_nf = np.load('D:\\project_python\\STG-NF\\save_path\\shanghaitech\\shanghaitech_5\\normality_scores_ep4.npy',allow_pickle=True)
pose_nf = np.load('utils\\scores_nfpose.npy',allow_pickle=True)
apose = np.load('utils\\final_scores_pose.npy',allow_pickle=True)
# deep = np.load('utils\\final_scores.npy',allow_pickle=True) # ###flow 试试超参数
final_score = - optical_nf - pose_nf # 0.8143640082873005  失败，两个精度差别太多, 没有互补，而是和vel可以互补


auc = roc_auc_score(gt, final_score)
print(auc)
# 归一化 vel
# vel =( vel - vel.min()) / (np.percentile(vel,99) - vel.min())
# vel = vel/(np.percentile(vel,99))
# vel = vel/vel.max()
# print(vel.shape)
