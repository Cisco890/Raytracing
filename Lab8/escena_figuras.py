import pygame
import numpy as np
from gl import Renderer
from figures import Sphere, Cube, Disk, Triangle, Plane, Capsule, Torus
from material import Material, OPAQUE, REFLECTIVE, TRANSPARENT
from lights import DirectionalLight, AmbientLight, PointLight
from camera import Camera

WIDTH, HEIGHT = 400, 400

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Raytracing - Cápsulas y Donas')

    renderer = Renderer(screen)
    renderer.camera = Camera()
    
    # Configurar recursión para reflexiones y transparencias
    renderer.maxRecursionDepth = 5
    
    # Sin fondo (negro)
    renderer.glClearColor(0.0, 0.0, 0.0)
    
    # No usar textura de fondo (comentado para crear cuarto cerrado)
    # renderer.envMap = None

    # Materiales para las figuras
    
    # Materiales reflectivos
    material_plateado = Material(diffuse=[0.9, 0.9, 0.9], spec=256, ks=0.95, matType=REFLECTIVE)
    material_dorado = Material(diffuse=[1.0, 0.84, 0.0], spec=256, ks=0.9, matType=REFLECTIVE)
    
    # Materiales opacos
    material_turquesa = Material(diffuse=[0.25, 0.88, 0.82], spec=32, ks=0.3, matType=OPAQUE)
    material_lila = Material(diffuse=[0.8, 0.6, 0.8], spec=32, ks=0.3, matType=OPAQUE)
    
    # Materiales transparentes
    material_rojo_transparente = Material(diffuse=[0.8, 0.2, 0.2], spec=64, ks=0.5, matType=TRANSPARENT, ior=1.5)
    material_naranja_transparente = Material(diffuse=[1.0, 0.6, 0.2], spec=64, ks=0.5, matType=TRANSPARENT, ior=1.4)
    
    print("Materiales creados para todas las figuras")

    # Materiales para el cuarto
    # Material para techo y suelo (blanca.bmp)
    try:
        from BMPTexture import BMPTexture
        blanca_texture = BMPTexture("blanca.bmp")
        material_techo_suelo = Material(diffuse=[0.9, 0.9, 0.9], spec=16.0, ks=0.1, matType=OPAQUE, texture=blanca_texture)
        print("Textura blanca cargada para techo y suelo")
    except:
        print("No se pudo cargar blanca.bmp, usando material blanco sólido")
        material_techo_suelo = Material(diffuse=[0.9, 0.9, 0.9], spec=16.0, ks=0.1, matType=OPAQUE)
    
    # Material para paredes (quarzo.bmp)
    try:
        quarzo_wall_texture = BMPTexture("quarzo.bmp")
        material_paredes = Material(diffuse=[0.8, 0.8, 0.8], spec=16.0, ks=0.1, matType=OPAQUE, texture=quarzo_wall_texture)
        print("Textura quarzo cargada para paredes")
    except:
        print("No se pudo cargar quarzo.bmp, usando material gris sólido para paredes")
        material_paredes = Material(diffuse=[0.8, 0.8, 0.8], spec=16.0, ks=0.1, matType=OPAQUE)

    # Crear el cuarto cerrado con 5 planos
    
    # SUELO - Plano horizontal inferior (normal apunta hacia arriba)
    suelo = Plane(position=[0.0, -3.0, -5.0], normal=[0, 1, 0], material=material_techo_suelo)
    renderer.scene.append(suelo)
    
    # TECHO - Plano horizontal superior (normal apunta hacia abajo)
    techo = Plane(position=[0.0, 3.0, -5.0], normal=[0, -1, 0], material=material_techo_suelo)
    renderer.scene.append(techo)
    
    # PARED TRASERA - Plano vertical al fondo (normal apunta hacia adelante)
    pared_trasera = Plane(position=[0.0, 0.0, -8.0], normal=[0, 0, 1], material=material_paredes)
    renderer.scene.append(pared_trasera)
    
    # PARED IZQUIERDA - Plano vertical a la izquierda (normal apunta hacia la derecha)
    pared_izquierda = Plane(position=[-4.0, 0.0, -5.0], normal=[1, 0, 0], material=material_paredes)
    renderer.scene.append(pared_izquierda)
    
    # PARED DERECHA - Plano vertical a la derecha (normal apunta hacia la izquierda)
    pared_derecha = Plane(position=[4.0, 0.0, -5.0], normal=[-1, 0, 0], material=material_paredes)
    renderer.scene.append(pared_derecha)

    # Crear las figuras según especificaciones
    
    # GRUPO 1: Cápsula plateada reflectiva y dona dorada reflectiva (izquierda)
    capsula_plateada = Capsule(position=[-2.5, -1.0, -5.0], radius=0.6, height=1.5, material=material_plateado)
    renderer.scene.append(capsula_plateada)
    
    dona_dorada = Torus(position=[-2.5, 1.2, -5.0], major_radius=0.8, minor_radius=0.3, material=material_dorado)
    renderer.scene.append(dona_dorada)
    
    # GRUPO 2: Cápsula turquesa opaca y dona lila opaca (centro)
    capsula_turquesa = Capsule(position=[0.0, -1.0, -5.0], radius=0.6, height=1.5, material=material_turquesa)
    renderer.scene.append(capsula_turquesa)
    
    dona_lila = Torus(position=[0.0, 1.2, -5.0], major_radius=0.8, minor_radius=0.3, material=material_lila)
    renderer.scene.append(dona_lila)
    
    # GRUPO 3: Cápsula roja transparente y dona naranja transparente (derecha)
    capsula_roja = Capsule(position=[2.5, -1.0, -5.0], radius=0.6, height=1.5, material=material_rojo_transparente)
    renderer.scene.append(capsula_roja)
    
    dona_naranja = Torus(position=[2.5, 1.2, -5.0], major_radius=0.8, minor_radius=0.3, material=material_naranja_transparente)
    renderer.scene.append(dona_naranja)

    # Luces
    # Luz puntual principal - Como una bombilla en el techo del cuarto
    luz_interior = PointLight(
        color=[1.0, 0.95, 0.8],  # Luz cálida (ligeramente amarillenta)
        intensity=15.0,          # Intensidad alta para iluminar todo el cuarto
        position=[0.0, 2.5, -5.0]  # Posicionada en el centro superior del cuarto
    )
    renderer.lights.append(luz_interior)
    
    # Luz ambiental suave para iluminación base
    luz_ambiental = AmbientLight(color=[0.8, 0.8, 1.0], intensity=0.2)
    renderer.lights.append(luz_ambiental)
    
    # Luz direccional secundaria suave (opcional - simula luz indirecta)
    luz_indirecta = DirectionalLight(
        color=[0.6, 0.7, 1.0], 
        intensity=0.3, 
        direction=[0.5, -0.8, -0.3]
    )
    renderer.lights.append(luz_indirecta)

    print("Iniciando renderizado...")
    print("Escena creada con:")
    print("- Cuarto cerrado (suelo, techo y 3 paredes con texturas)")
    print("- GRUPO 1 (izquierda): Capsula plateada reflectiva + Dona dorada reflectiva")
    print("- GRUPO 2 (centro): Capsula turquesa opaca + Dona lila opaca")  
    print("- GRUPO 3 (derecha): Capsula roja transparente + Dona naranja transparente")
    print("- Iluminación interior realista")

    # Renderizar
    renderer.glRenderRaytracing()
    
    # Guardar la imagen
    try:
        pygame.image.save(screen, "escena_figuras_resultado.png")
        print("Imagen guardada como 'escena_figuras_resultado.png'")
    except:
        print("No se pudo guardar la imagen")

    print("Renderizado completado!")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    # Guardar imagen al presionar 'S'
                    try:
                        pygame.image.save(screen, "escena_figuras_resultado.png")
                        print("Imagen guardada!")
                    except:
                        print("Error al guardar imagen")
    
    pygame.quit()

if __name__ == '__main__':
    main()