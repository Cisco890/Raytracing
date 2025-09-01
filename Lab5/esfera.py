import pygame
import numpy as np
from gl import Renderer
from figures import Sphere
from material import Material
from lights import DirectionalLight
from camera import Camera

WIDTH, HEIGHT = 800, 600

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Raytracing Esfera')

    renderer = Renderer(screen)
    renderer.camera = Camera()
    
    renderer.glClearColor(0.0, 0.0, 0.0)

    # Crear materiales para diferentes colores
    material_rojo = Material(diffuse=[1.0, 0.0, 0.0], spec=32.0, ks=0.3)
    material_azul = Material(diffuse=[0.0, 0.0, 1.0], spec=32.0, ks=0.3)
    material_verde = Material(diffuse=[0.0, 1.0, 0.0], spec=32.0, ks=0.3)
    
    # 4 esferas rojas en el centro 
    esfera_roja1 = Sphere(position=[-0.1, 0.1, -2.8], radius=0.22, material=material_rojo)  
    esfera_roja2 = Sphere(position=[0.1, -0.1, -3.2], radius=0.22, material=material_rojo) 
    esfera_roja3 = Sphere(position=[0.05, 0.2, -3.0], radius=0.20, material=material_rojo) 
    esfera_roja4 = Sphere(position=[-0.05, -0.2, -3.0], radius=0.20, material=material_rojo) 
    renderer.scene.append(esfera_roja1)
    renderer.scene.append(esfera_roja2)
    renderer.scene.append(esfera_roja3)
    renderer.scene.append(esfera_roja4)

    # 5 esferas azules cerca del centro
    esfera_azul1 = Sphere(position=[0, 0.4, -2.7], radius=0.18, material=material_azul)
    esfera_azul2 = Sphere(position=[-0.35, -0.15, -3.3], radius=0.18, material=material_azul)
    esfera_azul3 = Sphere(position=[0.35, -0.15, -2.9], radius=0.18, material=material_azul)  
    esfera_azul4 = Sphere(position=[-0.25, 0.3, -3.1], radius=0.16, material=material_azul)  
    esfera_azul5 = Sphere(position=[0.25, 0.25, -3.1], radius=0.16, material=material_azul)  
    renderer.scene.append(esfera_azul1)
    renderer.scene.append(esfera_azul2)
    renderer.scene.append(esfera_azul3)
    renderer.scene.append(esfera_azul4)
    renderer.scene.append(esfera_azul5)
    
    # 5 esferas verdes en órbitas exteriores 
    esfera_verde1 = Sphere(position=[0, 1.3, -2.5], radius=0.14, material=material_verde)     
    esfera_verde2 = Sphere(position=[-1.1, -0.7, -3.5], radius=0.14, material=material_verde)       
    esfera_verde3 = Sphere(position=[1.1, -0.7, -2.8], radius=0.14, material=material_verde)  
    esfera_verde4 = Sphere(position=[-0.8, 1.0, -3.2], radius=0.13, material=material_verde)  
    esfera_verde5 = Sphere(position=[0.8, 1.0, -2.9], radius=0.13, material=material_verde)  
    renderer.scene.append(esfera_verde1)
    renderer.scene.append(esfera_verde2)
    renderer.scene.append(esfera_verde3)

    # Crear luz direccional para el modelo Phong
    luz_direccional = DirectionalLight(color=[1,1,1], intensity=0.8, direction=[1, -1, -1])
    renderer.lights.append(luz_direccional)
    
    # Agregar luz ambiental
    from lights import AmbientLight
    luz_ambiental = AmbientLight(color=[1,1,1], intensity=0.2)
    renderer.lights.append(luz_ambiental)

    # Renderizar
    renderer.glRenderRaytracing()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

if __name__ == '__main__':
    main()
