
import numpy as np
from math import pi, atan2, asin
from intecept import Intercept

class Shape(object):
	def __init__(self, position, material):
		self.position = position
		self.material = material
		self.type = "None"

	def ray_intersect(self, orig, dir):
		return None


class Sphere(Shape):
	def __init__(self, position, radius, material):
		super().__init__(position, material)
		self.radius = radius
		self.type = "Sphere"
    
	def ray_intersect(self, orig, dir):
		# Asegurarse de trabajar con numpy arrays
		orig = np.array(orig, dtype=float)
		dir = np.array(dir, dtype=float)

		# Vector desde el origen del rayo hasta el centro de la esfera
		dir_length = np.linalg.norm(dir)
		if dir_length == 0:
			return None
		dir = dir / dir_length  # Normalizar la dirección del rayo

		# Vector del origen del rayo al centro de la esfera
		origin_to_center = np.array(self.position, dtype=float) - orig

		# Proyeccion de que tan lejos esta el punto mas cercano del rayo al centro
		projection_distance = np.dot(origin_to_center, dir)

		# Distancia perpendicular al cuadrado (usando pitagoras)
		perpendicular_distance_squared = np.dot(origin_to_center, origin_to_center) - projection_distance ** 2
		radius_squared = self.radius * self.radius

		# Si el rayo pasa mas lejos que el radio, no hay interseccion
		if perpendicular_distance_squared > radius_squared:
			return None
        
		# Distanci desde el punto de proyeccion hasta las intersecciones
		half_chord_distance = np.sqrt(radius_squared - perpendicular_distance_squared)

		# Las 2 distancias de interseccion
		near_distance = projection_distance - half_chord_distance
		far_distance = projection_distance + half_chord_distance

		# Elegir la interseccion mas cercana que este adelante del origen
		epsilon = 1e-6
		if near_distance > epsilon:
			# Calcular punto de impacto y normal
			hit_point = orig + dir * near_distance
			normal = (hit_point - np.array(self.position)) / self.radius
			
			# Calcular coordenadas de textura UV para esfera
			# Clamp normal[1] to prevent math domain errors
			normal_y_clamped = max(-1.0, min(1.0, normal[1]))
			u = 0.5 + atan2(normal[2], normal[0]) / (2 * pi)
			v = 0.5 - asin(normal_y_clamped) / pi
			texCoords = [u, v]
			
			# Devolver Intercept con toda la info
			return Intercept(hit_point, normal, near_distance, self, dir, texCoords)
    
		if far_distance > epsilon:
			# Lo mismo para la intersección lejana
			hit_point = orig + dir * far_distance
			normal = (hit_point - np.array(self.position)) / self.radius
			
			# Calcular coordenadas de textura UV para esfera
			# Clamp normal[1] to prevent math domain errors
			normal_y_clamped = max(-1.0, min(1.0, normal[1]))
			u = 0.5 + atan2(normal[2], normal[0]) / (2 * pi)
			v = 0.5 - asin(normal_y_clamped) / pi
			texCoords = [u, v]
			
			return Intercept(hit_point, normal, far_distance, self, dir, texCoords)
        
		# Ambas estan detras del origen
		return None

	