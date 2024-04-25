import pickle

import numpy as np

root_path = 'output/kitti_models/second_iou_car/source_only_features/eval/epoch_72/val/'
# root_path = 'output/kitti_models/second_iou_car/general_w_aug_pretrained_t_o_pretrained_2_10_only_80/eval/epoch_80/val/'

path_64 = root_path + '64/all_features_first_100.pkl'
path_32 = root_path + '32/all_features_first_100.pkl'
path_32_ = root_path + '32^/all_features_first_100.pkl'
path_16 = root_path + '16/all_features_first_100.pkl'
path_16_ = root_path + '16^/all_features_first_100.pkl'

with open(path_64, 'rb') as f:
    all_features_64 = pickle.load(f)

with open(path_32, 'rb') as f:
    all_features_32 = pickle.load(f)

with open(path_32_, 'rb') as f:
    all_features_32_ = pickle.load(f)

with open(path_16, 'rb') as f:
    all_features_16 = pickle.load(f)

with open(path_16_, 'rb') as f:
    all_features_16_ = pickle.load(f)

# shared_features_64 = all_features_64['shared_features']
# shared_features_32 = all_features_32['shared_features']
# shared_features_32_ = all_features_32_['shared_features']
# shared_features_16 = all_features_16['shared_features']
# shared_features_16_ = all_features_16_['shared_features']

# shared_features_all = np.concatenate([shared_features_64, shared_features_32, shared_features_32_, shared_features_16, shared_features_16_], axis=0)
# shared_features_all = shared_features_all.reshape(-1, 256)
# print(shared_features_all.shape)

# tsne to visualize the features
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(shared_features_all)

# # Draw the plot with the labels
# idx_64 = shared_features_64.shape[0]
# idx_32 = idx_64 + shared_features_32.shape[0]
# idx_32_ = idx_32 + shared_features_32_.shape[0]
# idx_16 = idx_32_ + shared_features_16.shape[0]
# idx_16_ = idx_16 + shared_features_16_.shape[0]

# plt.scatter(tsne_results[:idx_64, 0], tsne_results[:idx_64, 1], label='64', s=1)
# plt.scatter(tsne_results[idx_64:idx_32, 0], tsne_results[idx_64:idx_32, 1], label='32', s=1)
# plt.scatter(tsne_results[idx_32:idx_32_, 0], tsne_results[idx_32:idx_32_, 1], label='32^', s=1)
# plt.scatter(tsne_results[idx_32_:idx_16, 0], tsne_results[idx_32_:idx_16, 1], label='16', s=1)
# plt.scatter(tsne_results[idx_16:idx_16_, 0], tsne_results[idx_16:idx_16_, 1], label='16^', s=1)

# plt.legend()

# plt.savefig('tsne_so.png')

# PCA to visualize the features
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# pca = PCA(n_components=2)
# pca_results = pca.fit_transform(shared_features_all)

# # Draw the plot with the labels
# idx_64 = shared_features_64.shape[0]
# idx_32 = idx_64 + shared_features_32.shape[0]
# idx_32_ = idx_32 + shared_features_32_.shape[0]
# idx_16 = idx_32_ + shared_features_16.shape[0]
# idx_16_ = idx_16 + shared_features_16_.shape[0]

# plt.scatter(pca_results[:idx_64, 0], pca_results[:idx_64, 1], label='64', s=1)
# plt.scatter(pca_results[idx_64:idx_32, 0], pca_results[idx_64:idx_32, 1], label='32', s=1)
# plt.scatter(pca_results[idx_32:idx_32_, 0], pca_results[idx_32:idx_32_, 1], label='32^', s=1)
# plt.scatter(pca_results[idx_32_:idx_16, 0], pca_results[idx_32_:idx_16, 1], label='16', s=1)
# plt.scatter(pca_results[idx_16:idx_16_, 0], pca_results[idx_16:idx_16_, 1], label='16^', s=1)

# plt.legend()

# plt.savefig('pca.png')
frame_id = all_features_64['frame_id']
bev_features_64 = all_features_16['bev_features']
bev_features_64 = bev_features_64.transpose(0, 2, 3, 1)
frames, h, w, c = bev_features_64.shape
print(bev_features_64.shape)

bev_feature = bev_features_64[10]

import matplotlib.pyplot as plt

# # Sum and normalize the feature map
# bev_feature_img = np.sum(bev_feature, axis=2)
# bev_feature_img = (bev_feature_img - np.min(bev_feature_img)) / (np.max(bev_feature_img) - np.min(bev_feature_img))

# # map color to the feature map
# bev_feature_img = plt.cm.viridis(bev_feature_img)
# plt.imshow(bev_feature_img)
# plt.savefig('bev_feature.png')

# Draw point clouds
# Map frame_id to six digits
frame_id_curr = frame_id[10]
frame_id_str = str(frame_id_curr)
frame_id_str = '0' * (6 - len(frame_id_str)) + frame_id_str
print(frame_id_str)

# Load the points (.bin)
points = np.fromfile(f'data/kitti/training/modes/64/{frame_id_str}.bin', dtype=np.float32).reshape(-1, 4)
print(points.shape)

import open3d as o3d

# Draw the point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
o3d.visualization.draw_geometries([pcd])

# Save image
o3d.io.write_image('pcd_64.png', pcd)
