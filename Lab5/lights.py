import numpy as np
from intecept import Intercept
from MathLib import reflectVector

class Light(object):
    def __init__(self, color=[1,1,1], intensity=1.0, LightType="none"):
        self.color = color
        self.intensity = intensity
        self.LightType = LightType

    def GetLightColor(self, intercept=None ):
        return [(i * self.intensity) for i in self.color]
    
    def GetSpecularColor(self, intercept, viewPos):
        return[0,0,0]
    

class DirectionalLight(Light):
    def __init__(self, color=[1,1,1], intensity=1.0, direction=[0,-1,0]):
        super().__init__(color, intensity, "Directional")
        self.direction = direction / np.linalg.norm(direction)

    def GetLightColor(self, intercept=None):
        lightColor = super().GetLightColor()

        if intercept:
            # surfaceIntensity = NORMAL o -LIGHT
            dir = [(i * -1) for i in self.direction]
            surfaceIntensity = np.dot(intercept.normal, dir)
            # surfaceIntensity *= 1 - intercept.obj.material.ks
            surfaceIntensity = max(0, min(1, surfaceIntensity))
            lightColor = [(i * surfaceIntensity) for i in lightColor]

        return lightColor
    
    def GetSpecularColor(self, intercept, viewPos):
        specColor = self.color 

        # R = L - 2 * (L . N) * N
        if intercept:
            dir = [(i * -1) for i in self.direction]
            reflect = reflectVector(intercept.normal, dir)

            #SpecIntensity = ((V . R)^spec)*ks 

            viewDir = np.subtract(viewPos, intercept.point)
            viewDir = viewDir / np.linalg.norm(viewDir)

            specIntensity = max(0, np.dot(viewDir, reflect)) ** intercept.obj.material.spec
            specIntensity *= intercept.obj.material.ks
            specIntensity *= self.intensity
            specIntensity = max(0,  specIntensity)
            specColor = [(i * specIntensity) for i in specColor]

        return specColor
    

class AmbientLight(Light):
    def __init__(self, color=[1,1,1], intensity=0.1):
        super().__init__(color, intensity, LightType="Ambient")

