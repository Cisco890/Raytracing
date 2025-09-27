import numpy as np
from intecept import Intercept
from MathLib import reflectVector
from math import cos, radians, pi

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

    def GetDiffuseColor(self, intercept):
        # Basic diffuse calculation: lambertian reflectance
        if intercept:
            dir = [(i * -1) for i in self.direction]
            diffuse_intensity = max(0, np.dot(intercept.normal, dir))
            diffuse_color = [c * diffuse_intensity * self.intensity for c in self.color]
            return diffuse_color
        return [0, 0, 0]
    

class AmbientLight(Light):
    def __init__(self, color=[1,1,1], intensity=0.1):
        super().__init__(color, intensity, LightType="Ambient")

    def GetDiffuseColor(self, intercept=None):
        # Ambient light contributes its color scaled by intensity
        return [c * self.intensity for c in self.color]


class PointLight(Light):
    def __init__(self, color=[1, 1, 1], intensity=1.0, position=[0, 0, 0]):
        super().__init__(color, intensity, "Point")
        self.position = position

    def GetLightColor(self, intercept=None):
        lightColor = super().GetLightColor(intercept)

        if intercept:
            dir = np.subtract(self.position, intercept.point)
            # Distancia R, distancia desde el punto de interseccion hasta la luz
            R = np.linalg.norm(dir)
            dir /= R  # Normalizar la direccion

            surfaceIntensity = np.dot(intercept.normal, dir)

            # Ley de cuadrados inversos
            # Atenuaacion = intensidad / R ^ 2

            if R != 0:
                surfaceIntensity /= R**2

            surfaceIntensity = max(0, min(1, surfaceIntensity))
            lightColor = [(i * surfaceIntensity) for i in lightColor]

        return lightColor
    
    def GetSpecularColor(self, intercept, viewPos):
        specColor = super().GetLightColor()

        # Reflejo = 2 * (N · L) * N - L

        if intercept:
            dir = np.subtract(self.position, intercept.point)
            # Distancia R, distancia desde el punto de interseccion hasta la luz
            R = np.linalg.norm(dir)
            dir /= R  # Normalizar la direccion

            reflect = reflectVector(intercept.normal, dir)

            # SpecIntensity = ( (V · R) ** spec ) * ks
            viewDir = np.subtract(viewPos, intercept.point)
            viewDir /= np.linalg.norm(viewDir)

            specIntensity = max(0, np.dot(viewDir, reflect)) ** intercept.obj.material.spec
            specIntensity *= intercept.obj.material.ks

            # Ley de cuadrados inversos
            # Atenuaacion = intensidad / R ^ 2

            if R != 0:
                specIntensity /= R**2

            specColor = [(i * specIntensity) for i in specColor]

        return specColor
    
    def GetDiffuseColor(self, intercept):
        # Diffuse calculation for point light with distance attenuation
        if intercept:
            dir = np.subtract(self.position, intercept.point)
            # Distancia R, distancia desde el punto de interseccion hasta la luz
            R = np.linalg.norm(dir)
            dir /= R  # Normalizar la direccion

            diffuse_intensity = max(0, np.dot(intercept.normal, dir))
            
            # Ley de cuadrados inversos
            # Atenuacion = intensidad / R ^ 2
            if R != 0:
                diffuse_intensity /= R**2
            
            diffuse_intensity *= self.intensity
            diffuse_color = [c * diffuse_intensity for c in self.color]
            return diffuse_color
        return [0, 0, 0]


class SpotLight(PointLight):
    def __init__(self, color=[1, 1, 1], intensity=1.0, position=[0, 0, 0], direction=[0, -1, 0], innerAngle=50, outerAngle=60):
        super().__init__(color, intensity, position)
        self.direction = direction / np.linalg.norm(direction)
        self.innerAngle = innerAngle
        self.outerAngle = outerAngle
        self.LightType = "Spot"

    def GetLightColor(self, intercept=None):
        lightColor = super().GetLightColor(intercept)
        lightColor = [(i * self.EdgeAttenuation(intercept)) for i in lightColor]
        return lightColor
    
    def GetSpecularColor(self, intercept, viewPos):
        specColor = super().GetSpecularColor(intercept, viewPos)
        specColor = [(i * self.EdgeAttenuation(intercept)) for i in specColor]
        return specColor
    
    def EdgeAttenuation(self, intercept=None):
        if intercept == None:
            return 0
        
        # wi = direccion del punto a la luz
        # EdgeAttenuation = (-(DIR · wi) - cos(outerAngle)) / (cos(innerAngle) - cos(outerAngle))

        wi = np.subtract(self.position, intercept.point)
        wi /= np.linalg.norm(wi)

        from math import cos, pi
        innerAngleRads = self.innerAngle * pi / 180
        outerAngleRads = self.outerAngle * pi / 180

        attenuation = (-(np.dot(self.direction, wi)) - cos(outerAngleRads)) / (cos(innerAngleRads) - cos(outerAngleRads))
        attenuation = max(0, min(1, attenuation))
        
        return attenuation
    
    def GetDiffuseColor(self, intercept):
        # Diffuse calculation for spot light with distance and edge attenuation
        diffuse_color = super().GetDiffuseColor(intercept)
        diffuse_color = [(i * self.EdgeAttenuation(intercept)) for i in diffuse_color]
        return diffuse_color