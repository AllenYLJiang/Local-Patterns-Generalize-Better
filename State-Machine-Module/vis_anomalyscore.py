import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
def macro_auc(video, test_labels, lengths):
    prev = 0
    auc = 0
    for i, cur in enumerate(lengths):
        cur_auc = roc_auc_score(np.concatenate(([0], test_labels[prev: cur], [1])),
                             np.concatenate(([0], video[prev: cur], [sys.float_info.max])))
        auc += cur_auc
        prev = cur
    return auc / len(lengths)

sys.path.append('D:\\project_python\\STG-NF\\utils_r')
root = "D:\project_python\Accurate-Interpretable-VAD\data"
flist = "D:/project_python/Accurate-Interpretable-VAD/data/shanghaitech/testing/frames"
groudt = np.load('utils_r\\test_gt.npy',allow_pickle=True)
# 把gt 的 0 1 颠倒
# gt = 1 - gt
vel = np.load('utils_r\\final_scores_velocity.npy',allow_pickle=True)
optical_nf = np.load('D:\\project_python\\STG-NF\\save_path\\shanghaitech\\shanghaitech_5\\normality_scores_ep4.npy',allow_pickle=True)
pose_nf = np.load('utils_r\\scores_nfpose.npy',allow_pickle=True)
apose = np.load('utils_r\\final_scores_pose.npy',allow_pickle=True)
s4_score = np.load('save_path\S4shanghai_d64_f20\\normality_scores_ep12.npy',allow_pickle=True) # 0为异常 越低越异常 
s4_reverse = np.load("save_path\\S4shanghai_reverse\\normality_scores_ep7.npy",allow_pickle=True)
deep = np.load('utils_r\\final_scores_deep.npy',allow_pickle=True)
person_p = np.load("D:\project_python\Accurate-Interpretable-VAD\models\clip\saved_res\shanghaitech\person_ornot.npy",allow_pickle=True)

# print("s4,",s4_score.mean(),s4_score.min(),s4_score.max(),"deep,",deep.mean(),deep.min(),deep.max())
test_clip_lengths = np.load(os.path.join(root, "shanghaitech", 'test_clip_lengths.npy'))
# 得到文件夹中文件夹的list
flist = [os.path.join(flist, f) for f in os.listdir(flist) if os.path.isdir(os.path.join(flist, f))]

##  ['walking man', 'person',  'skating boy','riding bicycle', 'riding e-bike', 'driving car'] 
person_p = np.load("D:\project_python\Accurate-Interpretable-VAD\models\clip\saved_res\shanghaitech\person_ornot.npy",allow_pickle=True)
# def guass_video(scores, lengths, sigma=5):
#     scores_g = np.zeros_like(scores)
#     prev = 0
#     for cur in lengths:
#         scores_g[prev: cur] = gaussian_filter1d(scores[prev: cur], sigma)
#         prev = cur
#     return scores_g

# # s4_score = guass_video(s4_score, test_clip_lengths, sigma=3)
# print(roc_auc_score(1-gt,  s4_score))
scores = -person_p

prev = 0
clip_i = 0
auc_list = []
for cur in test_clip_lengths:
    scores[prev: cur]
    # 一维数组的数据
    data = scores[prev: cur]
    gt = 1 - groudt[prev: cur]
    gt_1 = groudt[prev: cur]
    data_1 = - data
    # 创建横轴标签
    x_labels = range(len(data)//10 + 1)

    # # 创建纵轴标签
    # y_labels = range(1, max(data) + 1)

    # 绘制网格图
    plt.plot(data_1)
    plt.plot(gt_1)
    # 设置横轴和纵轴标签
    # plt.xticks(range(0, len(data), 10), range(1, len(data)+1, 10),fontsize=4)
    plt.xticks(range(0, len(data), 10), x_labels[::1],fontsize=4)
    # plt.yticks(y_labels)
    # 如果gt 全是 1
    if gt.all() == 0:
        gt[0] = 1
    auc = roc_auc_score(gt, data)
    auc_list.append(auc)
    # 添加标题和标签
    plt.title('Grid Plot' + str(auc))
    plt.xlabel('Index')
    plt.ylabel('Value')

    
    # 添加网格线
    plt.grid(True)
    # 保存图形到指定目录
    plt.savefig('save_path\clip_score_fig/shanghai_person_p/'+ flist[clip_i].split('\\')[-1] +'_grid_plot.png')
    # plt.show()
    plt.close()
    clip_i += 1
    prev = cur
print(np.mean(auc_list))