class Material(object):
	def __init__(self, diffuse = [1,1,1],spec= 1.0, ks = 0):
		self.diffuse = diffuse
		self.spec = spec
		self.ks = ks

	def GetSurfaceColor(self, intercept, renderer):
		lightColor = [0,0,0]
		specColor = [0,0,0]

		for light in renderer.lights:
			shadowIntercept = None

			if light.LightType == "Directional":
				lightDir = [-i for i in light.direction]
				shadowIntercept = renderer.glCastRay(intercept.point, lightDir, intercept.obj)

			if shadowIntercept == None:
				lightColor = [(lightColor[i] + light.GetLightColor(intercept)[i]) for i in range(3)]
				specColor = [(specColor[i] + light.GetSpecularColor(intercept, renderer.camera.translation )[i]) for i in range(3)]

		finalColor = [(self.diffuse[i] * lightColor[i] + specColor[i]) for i in range(3)]
		finalColor = [min(1, finalColor[i]) for i in range(3)]

		return finalColor
		