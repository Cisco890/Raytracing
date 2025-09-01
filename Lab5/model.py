from MathLib import *

class Model(object):
    def __init__(self):
        self.vertices = []       # flat list: x,y,z,x,y,z,...
        self.texcoords = []      # flat list: u,v,u,v,...
        self.faces = []          # list of faces with ((v_idx, vt_idx), (v_idx, vt_idx), (v_idx, vt_idx))
        self.translation = [0,0,0]
        self.rotation = [5,-75,5] 

        self.scale = [0.3,0.5,0.3]
        self.vertexShader = None

    def GetModelMatrix(self):
        translateMat = TranslationMatrix(self.translation[0],
                                         self.translation[1],
                                         self.translation[2])

        rotateMat = RotationMatrix(self.rotation[0],
                                   self.rotation[1],
                                   self.rotation[2])

        scaleMat = ScaleMatrix(self.scale[0],
                               self.scale[1],
                               self.scale[2])

        return translateMat * rotateMat * scaleMat
