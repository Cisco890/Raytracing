
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


class Ellipsoid(Shape):
	def __init__(self, position, radii, material):
		super().__init__(position, material)
		self.radii = np.array(radii, dtype=float)  # [rx, ry, rz]
		self.type = "Ellipsoid"
	
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
		
		# Transformar el rayo al espacio de la esfera unitaria
		# Dividir por los radios para escalar
		scaled_orig = (orig - np.array(self.position)) / self.radii
		scaled_dir = dir / self.radii
		
		# Ahora resolvemos la intersección con una esfera unitaria
		# usando la ecuación cuadrática: at^2 + bt + c = 0
		a = np.dot(scaled_dir, scaled_dir)
		b = 2.0 * np.dot(scaled_orig, scaled_dir)
		c = np.dot(scaled_orig, scaled_orig) - 1.0
		
		# Calcular el discriminante
		discriminant = b * b - 4 * a * c
		
		# Si el discriminante es negativo, no hay intersección
		if discriminant < 0:
			return None
		
		# Calcular las dos soluciones
		sqrt_discriminant = np.sqrt(discriminant)
		t1 = (-b - sqrt_discriminant) / (2 * a)
		t2 = (-b + sqrt_discriminant) / (2 * a)
		
		# Elegir la intersección más cercana que esté adelante del origen
		t = None
		if t1 > epsilon:
			t = t1
		elif t2 > epsilon:
			t = t2
		else:
			return None
		
		# Calcular el punto de impacto en el espacio original
		hit_point = orig + dir * t
		
		# Calcular la normal en el espacio del elipsoide
		# La normal se calcula como el gradiente de la función del elipsoide
		# Para un elipsoide: (x/rx)^2 + (y/ry)^2 + (z/rz)^2 = 1
		# El gradiente es: [2x/rx^2, 2y/ry^2, 2z/rz^2]
		relative_point = hit_point - np.array(self.position)
		normal = relative_point / (self.radii * self.radii)
		
		# Normalizar la normal
		normal_length = np.linalg.norm(normal)
		if normal_length > 0:
			normal = normal / normal_length
		
		# Calcular coordenadas de textura UV para el elipsoide
		# Normalizar el punto relativo por los radios para obtener las coordenadas esféricas
		normalized_point = relative_point / self.radii
		norm_length = np.linalg.norm(normalized_point)
		if norm_length > 0:
			normalized_point = normalized_point / norm_length
		
		# Clamp para evitar errores de dominio en funciones trigonométricas
		normalized_y = max(-1.0, min(1.0, normalized_point[1]))
		u = 0.5 + atan2(normalized_point[2], normalized_point[0]) / (2 * pi)
		v = 0.5 - asin(normalized_y) / pi
		texCoords = [u, v]
		
		return Intercept(hit_point, normal, t, self, dir, texCoords)


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
		
		# Mapear local_x y local_y a coordenadas UV (puedes ajustar el escalado según sea necesario)
		u = 0.5 + local_x * 0.1
		v = 0.5 + local_y * 0.1
		
		# Asegurar que las coordenadas UV estén en el rango [0, 1]
		u = max(0.0, min(1.0, u))
		v = max(0.0, min(1.0, v))
		
		return [u, v]
	
	def _intersect_cylinder(self, orig, dir, epsilon):
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



class Torus(Shape):
	def __init__(self, position, major_radius, minor_radius, material, axis='y'):
		super().__init__(position, material)
		self.major_radius = major_radius  # Radio mayor (desde el centro hasta el centro del tubo)
		self.minor_radius = minor_radius  # Radio menor (radio del tubo)
		# Eje del toro: 'y' (por defecto, agujero vertical) o 'z' (agujero mirando a la cámara)
		self.axis = axis
		self.type = "Torus"
	
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
		# Trasladar el rayo al sistema local del toro y, si es necesario, permutar ejes
		ro_world = orig - np.array(self.position)
		if self.axis == 'z':
			# Local: eje Y del toro = Z del mundo
			ro = np.array([ro_world[0], ro_world[2], ro_world[1]])
			dir_local = np.array([dir[0], dir[2], dir[1]])
		else:
			# Eje por defecto 'y'
			ro = ro_world
			dir_local = dir
		
		# Coeficientes para la ecuación cuártica del toro (en espacio local)
		R = self.major_radius
		r = self.minor_radius
		
		# Precálculos
		dot_ro_ro = np.dot(ro, ro)
		dot_rd_rd = np.dot(dir_local, dir_local)
		dot_ro_rd = np.dot(ro, dir_local)
		
		k = dot_ro_ro - R*R - r*r
		
		# Coeficientes de la ecuación cuártica: at^4 + bt^3 + ct^2 + dt + e = 0
		a = dot_rd_rd * dot_rd_rd
		b = 4.0 * dot_rd_rd * dot_ro_rd
		
		c = 2.0 * dot_rd_rd * k + 4.0 * dot_ro_rd * dot_ro_rd + 4.0 * R*R * dir_local[1]*dir_local[1]
		
		d = 4.0 * k * dot_ro_rd + 8.0 * R*R * ro[1] * dir_local[1]
		
		e = k*k + 4.0 * R*R * (ro[1]*ro[1] - r*r)
		
		# Resolver la ecuación cuártica usando el método de Ferrari
		roots = self._solve_quartic(a, b, c, d, e)
		
		# Encontrar la intersección más cercana que esté adelante del origen
		min_t = float('inf')
		closest_hit = None
		
		for t in roots:
			if t > epsilon and t < min_t:
				# Punto de impacto en espacio local
				hit_local = ro + dir_local * t
				# Normal en espacio local
				normal_local = self._calculate_torus_normal_local(hit_local)
				if normal_local is None:
					continue
				# Volver a espacio mundo
				if self.axis == 'z':
					hit_point = np.array(self.position) + np.array([hit_local[0], hit_local[2], hit_local[1]])
					normal = np.array([normal_local[0], normal_local[2], normal_local[1]])
				else:
					hit_point = np.array(self.position) + hit_local
					normal = normal_local
				# Normalizar normal en mundo
				nlen = np.linalg.norm(normal)
				if nlen > 0:
					normal = normal / nlen
				# UV usando coords locales
				texCoords = self._calculate_uv_torus_local(hit_local)
				min_t = t
				closest_hit = Intercept(hit_point, normal, t, self, dir, texCoords)
		
		return closest_hit
	
	def _solve_quartic(self, a, b, c, d, e):
		roots = []
		
		# Normalizar coeficientes
		if abs(a) < 1e-10:
			return []
		
		b /= a
		c /= a
		d /= a
		e /= a
		
		# Método simplificado: buscar raíces usando aproximaciones numéricas
		# Evaluamos la función en varios puntos para encontrar cambios de signo
		def f(t):
			return t*t*t*t + b*t*t*t + c*t*t + d*t + e
		
		def df(t):
			return 4*t*t*t + 3*b*t*t + 2*c*t + d
		
		# Buscar raíces en un rango razonable
		test_points = np.linspace(-10, 10, 1000)
		
		for i in range(len(test_points) - 1):
			t1, t2 = test_points[i], test_points[i + 1]
			f1, f2 = f(t1), f(t2)
			
			# Si hay cambio de signo, hay una raíz
			if f1 * f2 < 0:
				# Usar método de Newton-Raphson para refinar la raíz
				t = (t1 + t2) / 2
				for _ in range(10):  # Máximo 10 iteraciones
					ft = f(t)
					dft = df(t)
					if abs(dft) < 1e-10:
						break
					t_new = t - ft / dft
					if abs(t_new - t) < 1e-8:
						break
					t = t_new
				
				if abs(f(t)) < 1e-6:  # Verificar que es una raíz válida
					roots.append(t)
		
		return roots
	
	def _calculate_torus_normal_local(self, p_local):
		R = self.major_radius
		r = self.minor_radius
		x, y, z = p_local[0], p_local[1], p_local[2]
		rho = np.sqrt(x*x + z*z)
		if rho < 1e-10:
			return None
		nx = 4.0 * x * (rho - R)
		ny = 4.0 * y * (rho - R) / rho if rho > 1e-10 else 0
		nz = 4.0 * z * (rho - R)
		normal = np.array([nx, ny, nz])
		nlen = np.linalg.norm(normal)
		if nlen < 1e-10:
			return None
		return normal / nlen
	
	def _calculate_uv_torus_local(self, p_local):
		x, y, z = p_local[0], p_local[1], p_local[2]
		u = 0.5 + atan2(z, x) / (2 * pi)
		rho = np.sqrt(x*x + z*z)
		if rho < 1e-10:
			v = 0.5
		else:
			circle_point = np.array([x * self.major_radius / rho, 0, z * self.major_radius / rho])
			to_hit = p_local - circle_point
			v = 0.5 + atan2(y, np.linalg.norm([to_hit[0], to_hit[2]]) - self.major_radius) / (2 * pi)
		u = u % 1.0
		v = v % 1.0
		if u < 0:
			u += 1.0
		if v < 0:
			v += 1.0
		return [u, v]


class Cylinder(Shape):
	def __init__(self, position, radius, height, material):
		super().__init__(position, material)
		self.radius = radius
		self.height = height
		self.type = "Cylinder"
		
		# Calcular las posiciones de las tapas superior e inferior
		half_height = self.height / 2
		self.bottom_y = position[1] - half_height
		self.top_y = position[1] + half_height
	
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
		
		# Trasladar el origen del rayo al sistema de coordenadas del cilindro
		oc = orig - np.array(self.position)
		
		# Para un cilindro vertical (eje Y), ignoramos la componente Y
		# Resolver la intersección con el cilindro infinito
		a = dir[0] * dir[0] + dir[2] * dir[2]
		b = 2.0 * (oc[0] * dir[0] + oc[2] * dir[2])
		c = oc[0] * oc[0] + oc[2] * oc[2] - self.radius * self.radius
		
		discriminant = b * b - 4 * a * c
		
		# Si no hay intersección con el cilindro infinito
		if discriminant < 0:
			return None
		
		sqrt_discriminant = np.sqrt(discriminant)
		t1 = (-b - sqrt_discriminant) / (2 * a) if abs(a) > epsilon else float('inf')
		t2 = (-b + sqrt_discriminant) / (2 * a) if abs(a) > epsilon else float('inf')
		
		closest_intercept = None
		closest_t = float('inf')
		
		# Verificar intersección con el cuerpo del cilindro
		for t in [t1, t2]:
			if t > epsilon:
				hit_point = orig + dir * t
				y = hit_point[1]
				
				# Verificar si está dentro de los límites de altura
				if self.bottom_y <= y <= self.top_y:
					if t < closest_t:
						# Calcular la normal (perpendicular al eje Y)
						center_point = np.array([self.position[0], y, self.position[2]])
						normal = (hit_point - center_point) / self.radius
						
						# Calcular coordenadas UV para el cuerpo del cilindro
						angle = atan2(normal[2], normal[0])
						u = 0.5 + angle / (2 * pi)
						v = (y - self.bottom_y) / self.height
						texCoords = [u, v]
						
						closest_t = t
						closest_intercept = Intercept(hit_point, normal, t, self, dir, texCoords)
		
		# Verificar intersección con la tapa inferior (disco en bottom_y)
		if abs(dir[1]) > epsilon:
			t_bottom = (self.bottom_y - orig[1]) / dir[1]
			if t_bottom > epsilon and t_bottom < closest_t:
				hit_point = orig + dir * t_bottom
				dx = hit_point[0] - self.position[0]
				dz = hit_point[2] - self.position[2]
				distance_squared = dx * dx + dz * dz
				
				if distance_squared <= self.radius * self.radius:
					normal = np.array([0, -1, 0])
					if np.dot(normal, dir) > 0:
						normal = -normal
					
					# Coordenadas UV para la tapa
					u = 0.5 + dx / (2 * self.radius)
					v = 0.5 + dz / (2 * self.radius)
					texCoords = [u, v]
					
					closest_t = t_bottom
					closest_intercept = Intercept(hit_point, normal, t_bottom, self, dir, texCoords)
		
		# Verificar intersección con la tapa superior (disco en top_y)
		if abs(dir[1]) > epsilon:
			t_top = (self.top_y - orig[1]) / dir[1]
			if t_top > epsilon and t_top < closest_t:
				hit_point = orig + dir * t_top
				dx = hit_point[0] - self.position[0]
				dz = hit_point[2] - self.position[2]
				distance_squared = dx * dx + dz * dz
				
				if distance_squared <= self.radius * self.radius:
					normal = np.array([0, 1, 0])
					if np.dot(normal, dir) > 0:
						normal = -normal
					
					# Coordenadas UV para la tapa
					u = 0.5 + dx / (2 * self.radius)
					v = 0.5 + dz / (2 * self.radius)
					texCoords = [u, v]
					
					closest_t = t_top
					closest_intercept = Intercept(hit_point, normal, t_top, self, dir, texCoords)
		
		return closest_intercept


class Pentagon(Shape):
	def __init__(self, position, radius, normal, material):
		super().__init__(position, material)
		self.radius = radius
		self.normal = np.array(normal, dtype=float)
		# Normalizar la normal del pentágono
		normal_length = np.linalg.norm(self.normal)
		if normal_length > 0:
			self.normal = self.normal / normal_length
		self.type = "Pentagon"
		
		# Calcular los vértices del pentágono
		self.vertices = self._calculate_vertices()
		
		# Crear 5 triángulos que forman el pentágono
		# Cada triángulo va desde el centro a dos vértices consecutivos
		self.triangles = []
		center = np.array(self.position)
		for i in range(5):
			v1 = self.vertices[i]
			v2 = self.vertices[(i + 1) % 5]  # El siguiente vértice (circular)
			# Crear un triángulo con el centro y dos vértices consecutivos
			triangle = Triangle(center, v1, v2, material)
			self.triangles.append(triangle)
	
	def _calculate_vertices(self):
		# Crear un sistema de coordenadas local
		z_axis = self.normal
		
		# Crear un eje X local arbitrario perpendicular a la normal
		if abs(z_axis[0]) < 0.9:
			x_axis = np.cross(z_axis, np.array([1, 0, 0]))
		else:
			x_axis = np.cross(z_axis, np.array([0, 1, 0]))
		x_axis = x_axis / np.linalg.norm(x_axis)
		
		# Crear el eje Y local usando el producto cruz
		y_axis = np.cross(z_axis, x_axis)
		
		# Calcular los 5 vértices del pentágono regular
		vertices = []
		num_sides = 5
		for i in range(num_sides):
			angle = 2 * pi * i / num_sides
			# Coordenadas en el sistema local
			local_x = self.radius * np.cos(angle)
			local_y = self.radius * np.sin(angle)
			
			# Transformar al espacio 3D
			vertex = (np.array(self.position) + 
					 local_x * x_axis + 
					 local_y * y_axis)
			vertices.append(vertex)
		
		return vertices
	
	def ray_intersect(self, orig, dir):
		# Asegurarse de trabajar con numpy arrays
		orig = np.array(orig, dtype=float)
		dir = np.array(dir, dtype=float)
		
		# Normalizar la dirección del rayo
		dir_length = np.linalg.norm(dir)
		if dir_length == 0:
			return None
		dir = dir / dir_length
		
		closest_intercept = None
		closest_distance = float('inf')
		
		# Probar intersección con cada uno de los 5 triángulos
		for triangle in self.triangles:
			intercept = triangle.ray_intersect(orig, dir)
			if intercept is not None:
				# Si encontramos una intersección, verificar si es la más cercana
				if intercept.distance < closest_distance:
					closest_distance = intercept.distance
					closest_intercept = intercept
					# Actualizar el objeto para que sea el pentágono, no el triángulo
					closest_intercept.obj = self
		
		return closest_intercept

	