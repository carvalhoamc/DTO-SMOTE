import numpy as np
from sklearn.decomposition import PCA

#geometry
order = ['area',#ok
         'volume',#ok
         'area_volume_ratio',#ok
         'edge_ratio',#ok
         'radius_ratio',#ok
         'aspect_ratio',#ok
         'max_solid_angle',
         'min_solid_angle',
         'solid_angle']

# Dirichlet Distribution alphas
alphas = np.arange(1,10,0.5)

# Compression Methods
projectors = [PCA()]

train_smote_ext = ["_train", "_SMOTE", "_Borderline1", "_Borderline2", "_smoteSVM","_Geometric_SMOTE"]






