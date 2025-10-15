
from MathLib import *
import numpy as np
from refractionFunctions import refractVector, totalInternalReflection, fresnel
OPAQUE = 0
REFLECTIVE = 1
TRANSPARENT = 2

class Material(object):
	def __init__(self, diffuse = [1,1,1],spec= 1.0,ior = 1.0, ks = 0,	matType = OPAQUE, texture = None):
		self.diffuse = diffuse
		self.spec = spec
		self.ks = ks
		self.matType = matType
		self.ior = ior
		self.texture = texture

	def GetSurfaceColor(self, intercept, renderer, recursion = 0 ):
		lightColor = [0,0,0]
		specColor = [0,0,0]
		reflectColor = [0,0,0]

		for light in renderer.lights:

			shadowIntercept = None

			if light.LightType == "Directional":
				lightDir = [-i for i in light.direction]
				shadowIntercept = renderer.glCastRay(intercept.point, lightDir, intercept.obj)

			if shadowIntercept == None:
				specColor = [(specColor[i] + light.GetSpecularColor(intercept, renderer.camera.translation )[i]) for i in range(3)]

				if self.matType == OPAQUE:
					lightColor = [(lightColor[i] + light.GetDiffuseColor(intercept)[i]) for i in range(3)]


		if self.matType == REFLECTIVE:
			# Set a recursion limit, default to 3 if not present
			maxRecursion = getattr(renderer, 'maxRecursionDepth', 3)
			if recursion >= maxRecursion:
				reflectColor = renderer.ClearColor if hasattr(renderer, 'ClearColor') else [0,0,0]
			else:
				rayDir = [-i for i in intercept.rayDirection]
				reflect = reflectVector(intercept.normal, rayDir)
				
				# Agregar bias para evitar auto-intersección
				bias = [i * 0.001 for i in intercept.normal]
				reflectOrig = [intercept.point[i] + bias[i] for i in range(3)]
				
				reflectIntercept = renderer.glCastRay(reflectOrig, reflect, intercept.obj, recursion + 1)
				if reflectIntercept is not None:
					reflectColor = reflectIntercept.obj.material.GetSurfaceColor(reflectIntercept, renderer, recursion + 1)
				else:
					reflectColor = renderer.glEnvMapColor(intercept.point, reflect)

		elif self.matType == TRANSPARENT:
			# Revisar si el rayo entra o sale del material
			outside = np.dot(intercept.normal, intercept.rayDirection) < 0 

			#Agregar el Bias
			bias = [i *0.001 for i in intercept.normal]

			rayDir = [-i for i in intercept.rayDirection]
			reflect = reflectVector(intercept.normal, rayDir)
			reflectOrig = np.add(intercept.point, bias) if outside else np.subtract(intercept.point, bias)
			reflectIntercept = renderer.glCastRay(reflectOrig, reflect, None, recursion + 1)
			if reflectIntercept is not None:
				reflectColor = reflectIntercept.obj.material.GetSurfaceColor(reflectIntercept, renderer, recursion + 1)
			else:
				reflectColor = renderer.glEnvMapColor(intercept.point, reflect)

			if not totalInternalReflection(intercept.normal, intercept.rayDirection, 1.0, intercept.obj.material.ior):
				refract = refractVector(intercept.normal, intercept.rayDirection, 1.0, intercept.obj.material.ior)
				refractOrig = np.subtract(intercept.point, bias) if outside else np.add(intercept.point, bias)
				refractIntercept = renderer.glCastRay(refractOrig, refract, None, recursion + 1)

				if refractIntercept is not None:
					refractColor = refractIntercept.obj.material.GetSurfaceColor(refractIntercept, renderer, recursion + 1)
				else:
					refractColor = renderer.glEnvMapColor(intercept.point, refract)

				kr, kt = fresnel(intercept.normal, intercept.rayDirection, 1.0, intercept.obj.material.ior)
				reflectColor = [reflectColor[i] * kr for i in range(3)]
				refractColor = [refractColor[i] * kt for i in range(3)]


		# Si no existe refractColor, inicialízalo en [0,0,0]
		if 'refractColor' not in locals():
			refractColor = [0, 0, 0]

		finalColor = [self.diffuse[i] * (lightColor[i] + reflectColor[i] + refractColor[i]) for i in range(3)]
		finalColor = [finalColor[i] + specColor[i] for i in range(3)]
		finalColor = [min(1, finalColor[i]) for i in range(3)]

		# Apply texture if available
		if self.texture and hasattr(intercept, 'texCoords') and intercept.texCoords:
			textureColor = self.texture.getColor(intercept.texCoords[0], intercept.texCoords[1])
			finalColor = [finalColor[i] * textureColor[i] for i in range(3)]

		return finalColor
		