

class Intercept(object):
    def __init__(self, point, normal,distance,obj, rayDirection, texCoords = None):
        self.point = point
        self.normal = normal
        self.distance = distance
        self.rayDirection = rayDirection   
        self.obj = obj
        self.texCoords = texCoords

        #cambiar en el algorito para que en lugar de regresar falso o verdadero regrese un intercepto