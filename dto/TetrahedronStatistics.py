from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


class tetrahedron_stats:
	
	def __init__(self, pts):
		self.labels = "ABCD"
		self.hull = ConvexHull(pts)
		self.pts = dict(zip(self.labels, pts))
		self.dists = pdist(pts)
		self.angles = self.tetrahedron_solid_angles()
	
	def area(self):
		return self.hull.area
	
	def volume(self):
		return self.hull.volume
	
	def edge_ratio(self):
		return max(self.dists) / min(self.dists)
	
	def radius_ratio(self):
		circumcenter = self.calc_circumcenter()
		innercenter = self.calc_incencenter()
		return (3 * innercenter['radius'] / circumcenter['radius'])
	
	def aspect_ratio(self):
		E = max(self.dists)
		R = self.calc_circumcenter()['radius']
		return (R / E)
	
	def tetrahedron_edges(self):
		return ({"".join([i, j]): self.pts[i] - self.pts[j] for i, j in combinations(self.labels, 2)})
	
	def normals(self, edges):
		pairs = [("".join((l[0], l[1])), "".join((l[0], l[2]))) for l in combinations(self.labels, 3)]
		normals = [np.cross(edges[e1], edges[e2]) for (e1, e2) in pairs]
		return (dict(zip(self.labels, normals)))
	
	def angle(self, u, v):
		dot = np.dot(u, v)
		cos_theta = dot / (np.linalg.norm(u) * np.linalg.norm(v))
		return (np.arccos(cos_theta))
	
	def tetrahedron_dihedral_angles(self):
		edges = self.tetrahedron_edges()
		normals = self.normals(edges)
		angles = {"".join([i, j]): np.pi - self.angle(normals[i], normals[j]) for i, j in combinations(self.labels, 2)}
		return angles
	
	def tetrahedron_solid_angles(self):
		dihedral_angles = self.tetrahedron_dihedral_angles()
		triangles = [["".join(edge) for edge in combinations(facet, 2)] for facet in combinations(self.labels, 3)]
		return [np.sum([dihedral_angles[edge] for edge in triangle]) - np.pi for triangle in triangles]
	
	def max_solid_angle(self):
		return (max(self.angles))
	
	def min_solid_angle(self):
		return (min(self.angles))
	
	def solid_angle(self):
		return (sum(self.angles))
	
	def calc_circumcenter(self):
		""" Calculates the cirumcenters of the circumspheres of tetrahedrons.
		An implementation based on
		http://mathworld.wolfram.com/Circumsphere.html
		"""
		tetrahedron = np.array(list(self.pts.values()))
		
		a = np.concatenate((tetrahedron, np.ones((4, 1))), axis=1)
		
		sums = np.sum(tetrahedron ** 2, axis=1)
		d = np.concatenate((sums[:, np.newaxis], a), axis=1)
		c = np.concatenate((sums[:, np.newaxis], tetrahedron), axis=1)
		
		dx = np.delete(d, 1, axis=1)
		dy = np.delete(d, 2, axis=1)
		dz = np.delete(d, 3, axis=1)
		
		dx = np.linalg.det(dx)
		dy = -np.linalg.det(dy)
		dz = np.linalg.det(dz)
		a = np.linalg.det(a)
		c = np.linalg.det(c)
		
		nominator = np.array((dx, dy, dz))
		denominator = 2 * a
		
		radius = np.sqrt(sum(nominator ** 2) - 4 * a * c) / np.abs(denominator)
		
		return {"center": (nominator / denominator).T, "radius": radius}
	
	def calc_incencenter(self):
		tetrahedron = np.array(list(self.pts.values()))
		edges = self.tetrahedron_edges()
		normals = self.normals(edges)
		
		n = np.array(list(normals.values()))
		sums = np.sum(n ** 2, axis=1)
		A = np.concatenate((n, sums[:, np.newaxis]), axis=1)
		
		u = np.sum(np.multiply(n, tetrahedron), axis=1)
		
		s = np.linalg.solve(A, u)
		
		return {"center": s[0:3], "radius": np.abs(s[3])}
	
	def values(self):
		return ({'area': self.area(),
				 'volume': self.volume(),
				 'area_volume_ratio': self.volume() / self.area(),
				 'edge_ratio': self.edge_ratio(),
				 'radius_ratio': self.radius_ratio(),
				 'aspect_ratio': self.aspect_ratio(),
				 'max_solid_angle': self.max_solid_angle(),
				 'min_solid_angle': self.min_solid_angle(),
				 'solid_angle': self.solid_angle()
				 })
