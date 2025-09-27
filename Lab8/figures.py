
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


class Cube(Shape):
	def __init__(self, position, size, material):
		super().__init__(position, material)
		self.size = size  # Tamaño del cubo (lado)
		self.type = "Cube"
		
		# Calcular los límites del cubo
		half_size = self.size / 2
		self.min_bounds = np.array(self.position) - half_size
		self.max_bounds = np.array(self.position) + half_size
	
	def ray_intersect(self, orig, dir):
		# Asegurarse de trabajar con numpy arrays
		orig = np.array(orig, dtype=float)
		dir = np.array(dir, dtype=float)
		
		# Normalizar la dirección del rayo
		dir_length = np.linalg.norm(dir)
		if dir_length == 0:
			return None
		dir = dir / dir_length
		
		epsilon = 1e-6
		
		# Calcular las intersecciones con cada par de planos paralelos
		t_min = float('-inf')
		t_max = float('inf')
		normal_at_min = None
		
		for i in range(3):  # x, y, z
			if abs(dir[i]) < epsilon:
				# Rayo paralelo a los planos
				if orig[i] < self.min_bounds[i] or orig[i] > self.max_bounds[i]:
					return None
			else:
				# Calcular intersecciones con los planos
				t1 = (self.min_bounds[i] - orig[i]) / dir[i]
				t2 = (self.max_bounds[i] - orig[i]) / dir[i]
				
				# Asegurar que t1 <= t2
				if t1 > t2:
					t1, t2 = t2, t1
				
				# Actualizar t_min y t_max
				if t1 > t_min:
					t_min = t1
					# Determinar la normal en el punto de entrada
					normal_at_min = np.zeros(3)
					if abs(orig[i] + dir[i] * t_min - self.min_bounds[i]) < epsilon:
						normal_at_min[i] = -1.0
					else:
						normal_at_min[i] = 1.0
				
				if t2 < t_max:
					t_max = t2
			
			# Si t_min > t_max, no hay intersección
			if t_min > t_max:
				return None
		
		# Elegir la intersección más cercana que esté adelante del origen
		if t_min > epsilon:
			hit_point = orig + dir * t_min
			normal = normal_at_min
			
			# Calcular coordenadas de textura UV para el cubo
			texCoords = self._calculate_uv(hit_point, normal)
			
			return Intercept(hit_point, normal, t_min, self, dir, texCoords)
		
		elif t_max > epsilon:
			hit_point = orig + dir * t_max
			
			# Calcular la normal en el punto de salida
			normal = np.zeros(3)
			for i in range(3):
				if abs(hit_point[i] - self.min_bounds[i]) < epsilon:
					normal[i] = -1.0
					break
				elif abs(hit_point[i] - self.max_bounds[i]) < epsilon:
					normal[i] = 1.0
					break
			
			# Calcular coordenadas de textura UV para el cubo
			texCoords = self._calculate_uv(hit_point, normal)
			
			return Intercept(hit_point, normal, t_max, self, dir, texCoords)
		
		return None
	
	def _calculate_uv(self, hit_point, normal):
		"""Calcula las coordenadas UV para el mapeo de texturas en el cubo"""
		# Normalizar el punto de impacto relativo al cubo
		relative_point = hit_point - np.array(self.position)
		half_size = self.size / 2
		
		# Determinar qué cara del cubo estamos mirando basado en la normal
		if abs(normal[0]) > 0.5:  # Cara X (izquierda/derecha)
			u = (relative_point[2] / half_size + 1) * 0.5
			v = (relative_point[1] / half_size + 1) * 0.5
		elif abs(normal[1]) > 0.5:  # Cara Y (arriba/abajo)
			u = (relative_point[0] / half_size + 1) * 0.5
			v = (relative_point[2] / half_size + 1) * 0.5
		else:  # Cara Z (frente/atrás)
			u = (relative_point[0] / half_size + 1) * 0.5
			v = (relative_point[1] / half_size + 1) * 0.5
		
		# Asegurar que las coordenadas UV estén en el rango [0, 1]
		u = max(0.0, min(1.0, u))
		v = max(0.0, min(1.0, v))
		
		return [u, v]


class Disk(Shape):
	def __init__(self, position, radius, normal, material):
		super().__init__(position, material)
		self.radius = radius
		self.normal = np.array(normal, dtype=float)
		# Normalizar la normal del disco
		normal_length = np.linalg.norm(self.normal)
		if normal_length > 0:
			self.normal = self.normal / normal_length
		self.type = "Disk"
	
	def ray_intersect(self, orig, dir):
		# Asegurarse de trabajar con numpy arrays
		orig = np.array(orig, dtype=float)
		dir = np.array(dir, dtype=float)
		
		# Normalizar la dirección del rayo
		dir_length = np.linalg.norm(dir)
		if dir_length == 0:
			return None
		dir = dir / dir_length
		
		epsilon = 1e-6
		
		# Calcular el producto punto entre la dirección del rayo y la normal del disco
		denom = np.dot(dir, self.normal)
		
		# Si el rayo es paralelo al disco, no hay intersección
		if abs(denom) < epsilon:
			return None
		
		# Vector del origen del rayo al centro del disco
		origin_to_center = np.array(self.position) - orig
		
		# Calcular la distancia t hasta el plano del disco
		t = np.dot(origin_to_center, self.normal) / denom
		
		# Si t es negativo, la intersección está detrás del origen del rayo
		if t < epsilon:
			return None
		
		# Calcular el punto de intersección
		hit_point = orig + dir * t
		
		# Verificar si el punto está dentro del radio del disco
		center_to_hit = hit_point - np.array(self.position)
		distance_squared = np.dot(center_to_hit, center_to_hit)
		
		if distance_squared > self.radius * self.radius:
			return None
		
		# Determinar la normal correcta (apuntando hacia el rayo)
		normal = self.normal
		if np.dot(normal, dir) > 0:
			normal = -normal
		
		# Calcular coordenadas de textura UV para el disco
		texCoords = self._calculate_uv_disk(hit_point)
		
		return Intercept(hit_point, normal, t, self, dir, texCoords)
	
	def _calculate_uv_disk(self, hit_point):
		"""Calcula las coordenadas UV para el mapeo de texturas en el disco"""
		# Vector del centro del disco al punto de impacto
		center_to_hit = hit_point - np.array(self.position)
		
		# Crear un sistema de coordenadas local para el disco
		# Usar la normal como eje Z local
		z_axis = self.normal
		
		# Crear un eje X local arbitrario perpendicular a la normal
		if abs(z_axis[0]) < 0.9:
			x_axis = np.cross(z_axis, np.array([1, 0, 0]))
		else:
			x_axis = np.cross(z_axis, np.array([0, 1, 0]))
		x_axis = x_axis / np.linalg.norm(x_axis)
		
		# Crear el eje Y local usando el producto cruz
		y_axis = np.cross(z_axis, x_axis)
		
		# Proyectar el vector center_to_hit en los ejes locales
		local_x = np.dot(center_to_hit, x_axis)
		local_y = np.dot(center_to_hit, y_axis)
		
		# Convertir a coordenadas polares y luego a UV
		distance = np.sqrt(local_x * local_x + local_y * local_y)
		angle = atan2(local_y, local_x)
		
		# Mapear la distancia al rango [0, 1] basado en el radio
		u = 0.5 + (distance / self.radius) * np.cos(angle) * 0.5
		v = 0.5 + (distance / self.radius) * np.sin(angle) * 0.5
		
		# Asegurar que las coordenadas UV estén en el rango [0, 1]
		u = max(0.0, min(1.0, u))
		v = max(0.0, min(1.0, v))
		
		return [u, v]


class Triangle(Shape):
	def __init__(self, v0, v1, v2, material):
		# Calcular el centro del triángulo como posición
		position = (np.array(v0) + np.array(v1) + np.array(v2)) / 3.0
		super().__init__(position, material)
		
		self.v0 = np.array(v0, dtype=float)
		self.v1 = np.array(v1, dtype=float)
		self.v2 = np.array(v2, dtype=float)
		self.type = "Triangle"
		
		# Pre-calcular la normal del triángulo
		edge1 = self.v1 - self.v0
		edge2 = self.v2 - self.v0
		self.normal = np.cross(edge1, edge2)
		normal_length = np.linalg.norm(self.normal)
		if normal_length > 0:
			self.normal = self.normal / normal_length
	
	def ray_intersect(self, orig, dir):
		# Asegurarse de trabajar con numpy arrays
		orig = np.array(orig, dtype=float)
		dir = np.array(dir, dtype=float)
		
		# Normalizar la dirección del rayo
		dir_length = np.linalg.norm(dir)
		if dir_length == 0:
			return None
		dir = dir / dir_length
		
		epsilon = 1e-6
		
		# Algoritmo de Möller-Trumbore para intersección ray-triangle
		edge1 = self.v1 - self.v0
		edge2 = self.v2 - self.v0
		
		# Calcular el determinante
		h = np.cross(dir, edge2)
		a = np.dot(edge1, h)
		
		# Si a está cerca de 0, el rayo es paralelo al triángulo
		if abs(a) < epsilon:
			return None
		
		f = 1.0 / a
		s = orig - self.v0
		u = f * np.dot(s, h)
		
		# Verificar si u está fuera del triángulo
		if u < 0.0 or u > 1.0:
			return None
		
		q = np.cross(s, edge1)
		v = f * np.dot(dir, q)
		
		# Verificar si v está fuera del triángulo
		if v < 0.0 or u + v > 1.0:
			return None
		
		# Calcular t para encontrar el punto de intersección
		t = f * np.dot(edge2, q)
		
		# Si t es positivo, hay intersección
		if t > epsilon:
			hit_point = orig + dir * t
			
			# Determinar la normal correcta (apuntando hacia el rayo)
			normal = self.normal
			if np.dot(normal, dir) > 0:
				normal = -normal
			
			# Calcular coordenadas de textura UV usando coordenadas baricéntricas
			# u y v ya calculados arriba son las coordenadas baricéntricas
			w = 1.0 - u - v  # Tercera coordenada baricéntrica
			
			# Mapear coordenadas baricéntricas a UV para textura
			texCoords = [u, v]  # Usar directamente las coordenadas baricéntricas
			
			return Intercept(hit_point, normal, t, self, dir, texCoords)
		
		return None


class Plane(Shape):
	def __init__(self, position, normal, material):
		super().__init__(position, material)
		self.normal = np.array(normal, dtype=float)
		# Normalizar la normal del plano
		normal_length = np.linalg.norm(self.normal)
		if normal_length > 0:
			self.normal = self.normal / normal_length
		self.type = "Plane"
	
	def ray_intersect(self, orig, dir):
		# Asegurarse de trabajar con numpy arrays
		orig = np.array(orig, dtype=float)
		dir = np.array(dir, dtype=float)
		
		# Normalizar la dirección del rayo
		dir_length = np.linalg.norm(dir)
		if dir_length == 0:
			return None
		dir = dir / dir_length
		
		epsilon = 1e-6
		
		# Calcular el producto punto entre la dirección del rayo y la normal del plano
		denom = np.dot(dir, self.normal)
		
		# Si el rayo es paralelo al plano, no hay intersección
		if abs(denom) < epsilon:
			return None
		
		# Vector del origen del rayo al punto del plano
		origin_to_plane = np.array(self.position) - orig
		
		# Calcular la distancia t hasta el plano
		t = np.dot(origin_to_plane, self.normal) / denom
		
		# Si t es negativo, la intersección está detrás del origen del rayo
		if t < epsilon:
			return None
		
		# Calcular el punto de intersección
		hit_point = orig + dir * t
		
		# Determinar la normal correcta (apuntando hacia el rayo)
		normal = self.normal
		if np.dot(normal, dir) > 0:
			normal = -normal
		
		# Calcular coordenadas de textura UV para el plano
		texCoords = self._calculate_uv_plane(hit_point)
		
		return Intercept(hit_point, normal, t, self, dir, texCoords)
	
	def _calculate_uv_plane(self, hit_point):
		"""Calcula las coordenadas UV para el mapeo de texturas en el plano"""
		# Vector del punto de referencia del plano al punto de impacto
		plane_to_hit = hit_point - np.array(self.position)
		
		# Crear un sistema de coordenadas local para el plano
		# Usar la normal como eje Z local
		z_axis = self.normal
		
		# Crear un eje X local arbitrario perpendicular a la normal
		if abs(z_axis[0]) < 0.9:
			x_axis = np.cross(z_axis, np.array([1, 0, 0]))
		else:
			x_axis = np.cross(z_axis, np.array([0, 1, 0]))
		x_axis = x_axis / np.linalg.norm(x_axis)
		
		# Crear el eje Y local usando el producto cruz
		y_axis = np.cross(z_axis, x_axis)
		
		# Proyectar el vector plane_to_hit en los ejes locales
		local_x = np.dot(plane_to_hit, x_axis)
		local_y = np.dot(plane_to_hit, y_axis)
		
		# Escalar para crear un patrón de textura repetitivo
		# Usar un factor de escala para controlar el tamaño del patrón
		scale_factor = 1.0  # Ajustar este valor para cambiar el tamaño del patrón
		u = (local_x * scale_factor) % 1.0
		v = (local_y * scale_factor) % 1.0
		
		# Asegurar que las coordenadas UV estén en el rango [0, 1]
		if u < 0:
			u += 1.0
		if v < 0:
			v += 1.0
		
		return [u, v]


class Capsule(Shape):
	def __init__(self, position, radius, height, material):
		super().__init__(position, material)
		self.radius = radius
		self.height = height  # Altura del cilindro central (sin contar las hemisferias)
		self.type = "Capsule"
		
		# Calcular los centros de las hemisferias superior e inferior
		half_height = self.height / 2
		self.top_center = np.array(self.position) + np.array([0, half_height, 0])
		self.bottom_center = np.array(self.position) - np.array([0, half_height, 0])
	
	def ray_intersect(self, orig, dir):
		# Asegurarse de trabajar con numpy arrays
		orig = np.array(orig, dtype=float)
		dir = np.array(dir, dtype=float)
		
		# Normalizar la dirección del rayo
		dir_length = np.linalg.norm(dir)
		if dir_length == 0:
			return None
		dir = dir / dir_length
		
		epsilon = 1e-6
		closest_intercept = None
		min_distance = float('inf')
		
		# 1. Intersección con el cilindro central
		cylinder_intercept = self._intersect_cylinder(orig, dir, epsilon)
		if cylinder_intercept and cylinder_intercept.distance < min_distance:
			min_distance = cylinder_intercept.distance
			closest_intercept = cylinder_intercept
		
		# 2. Intersección con la hemisfera superior
		top_sphere_intercept = self._intersect_hemisphere(orig, dir, self.top_center, True, epsilon)
		if top_sphere_intercept and top_sphere_intercept.distance < min_distance:
			min_distance = top_sphere_intercept.distance
			closest_intercept = top_sphere_intercept
		
		# 3. Intersección con la hemisfera inferior
		bottom_sphere_intercept = self._intersect_hemisphere(orig, dir, self.bottom_center, False, epsilon)
		if bottom_sphere_intercept and bottom_sphere_intercept.distance < min_distance:
			min_distance = bottom_sphere_intercept.distance
			closest_intercept = bottom_sphere_intercept
		
		return closest_intercept
	
	def _intersect_cylinder(self, orig, dir, epsilon):
		"""Intersección con la parte cilíndrica de la cápsula"""
		# El cilindro está alineado con el eje Y
		# Resolver la ecuación cuadrática para un cilindro infinito en Y
		a = dir[0] * dir[0] + dir[2] * dir[2]
		
		if abs(a) < epsilon:
			# Rayo paralelo al eje del cilindro
			distance_to_axis = np.sqrt((orig[0] - self.position[0])**2 + (orig[2] - self.position[2])**2)
			if distance_to_axis > self.radius:
				return None
		else:
			ox = orig[0] - self.position[0]
			oz = orig[2] - self.position[2]
			
			b = 2 * (dir[0] * ox + dir[2] * oz)
			c = ox * ox + oz * oz - self.radius * self.radius
			
			discriminant = b * b - 4 * a * c
			if discriminant < 0:
				return None
			
			sqrt_disc = np.sqrt(discriminant)
			t1 = (-b - sqrt_disc) / (2 * a)
			t2 = (-b + sqrt_disc) / (2 * a)
			
			# Verificar intersecciones válidas
			for t in [t1, t2]:
				if t > epsilon:
					hit_point = orig + dir * t
					
					# Verificar si el punto está dentro de los límites del cilindro
					half_height = self.height / 2
					if (self.position[1] - half_height) <= hit_point[1] <= (self.position[1] + half_height):
						# Calcular la normal
						normal = np.array([
							hit_point[0] - self.position[0],
							0,
							hit_point[2] - self.position[2]
						])
						normal = normal / np.linalg.norm(normal)
						
						# Calcular coordenadas UV
						texCoords = self._calculate_uv_cylinder(hit_point, normal)
						
						return Intercept(hit_point, normal, t, self, dir, texCoords)
		
		return None
	
	def _intersect_hemisphere(self, orig, dir, center, is_top, epsilon):
		"""Intersección con una hemisfera (superior o inferior)"""
		# Vector del origen del rayo al centro de la hemisfera
		origin_to_center = center - orig
		
		# Proyección de qué tan lejos está el punto más cercano del rayo al centro
		projection_distance = np.dot(origin_to_center, dir)
		
		# Distancia perpendicular al cuadrado
		perpendicular_distance_squared = np.dot(origin_to_center, origin_to_center) - projection_distance ** 2
		radius_squared = self.radius * self.radius
		
		# Si el rayo pasa más lejos que el radio, no hay intersección
		if perpendicular_distance_squared > radius_squared:
			return None
		
		# Distancia desde el punto de proyección hasta las intersecciones
		half_chord_distance = np.sqrt(radius_squared - perpendicular_distance_squared)
		
		# Las 2 distancias de intersección
		near_distance = projection_distance - half_chord_distance
		far_distance = projection_distance + half_chord_distance
		
		# Verificar intersecciones válidas
		for t in [near_distance, far_distance]:
			if t > epsilon:
				hit_point = orig + dir * t
				
				# Verificar si el punto está en la hemisfera correcta
				if is_top and hit_point[1] >= center[1]:
					normal = (hit_point - center) / self.radius
					texCoords = self._calculate_uv_hemisphere(hit_point, normal, is_top)
					return Intercept(hit_point, normal, t, self, dir, texCoords)
				elif not is_top and hit_point[1] <= center[1]:
					normal = (hit_point - center) / self.radius
					texCoords = self._calculate_uv_hemisphere(hit_point, normal, is_top)
					return Intercept(hit_point, normal, t, self, dir, texCoords)
		
		return None
	
	def _calculate_uv_cylinder(self, hit_point, normal):
		"""Calcula las coordenadas UV para la parte cilíndrica"""
		# Coordenada U basada en el ángulo alrededor del cilindro
		u = 0.5 + atan2(normal[2], normal[0]) / (2 * pi)
		
		# Coordenada V basada en la altura
		half_height = self.height / 2
		v = (hit_point[1] - (self.position[1] - half_height)) / self.height
		
		# Asegurar que las coordenadas UV estén en el rango [0, 1]
		u = max(0.0, min(1.0, u))
		v = max(0.0, min(1.0, v))
		
		return [u, v]
	
	def _calculate_uv_hemisphere(self, hit_point, normal, is_top):
		"""Calcula las coordenadas UV para las hemisferias"""
		# Para las hemisferias, usar coordenadas esféricas
		# Clamp normal[1] to prevent math domain errors
		normal_y_clamped = max(-1.0, min(1.0, normal[1]))
		
		u = 0.5 + atan2(normal[2], normal[0]) / (2 * pi)
		
		if is_top:
			# Para la hemisfera superior, mapear desde el ecuador (v=0.5) hasta el polo (v=1)
			v = 0.5 + (0.5 - asin(normal_y_clamped) / pi)
		else:
			# Para la hemisfera inferior, mapear desde el polo (v=0) hasta el ecuador (v=0.5)
			v = 0.5 + asin(normal_y_clamped) / pi
		
		# Asegurar que las coordenadas UV estén en el rango [0, 1]
		u = max(0.0, min(1.0, u))
		v = max(0.0, min(1.0, v))
		
		return [u, v]

	