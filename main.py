import numpy as np
from vis import visualize_prediction
import os

path = 'data/big_nyu'
point_cloud = np.load(os.path.join(path, 'point_cloud.npy'))
point_colors = np.load(os.path.join(path, 'point_colors.npy'))
poses = np.load(os.path.join(path, 'poses.npy'))

visualize_prediction(point_cloud, point_colors, poses)