from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

#geometry
order = ['area',
         'volume',
         'area_volume_ratio'
         'edge_ratio',
         'radius_ratio',
         'aspect_ratio',
         'max_solid_angle',
         'min_solid_angle',
         'solid_angle']

# Dirichlet Distribution alphas
alphas = [1,4,9]

# Compression Methods
projectors = [PCA()]








