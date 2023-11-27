import pygame
from collections import namedtuple


WHITE = (255, 255, 255)
RED1 = (200,0,0)
RED2 = (255, 100, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
BLACK = (0,0,0)

SPEED = 60

Point = namedtuple('Point', 'x, y')

class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class Display(Singleton):


    def __init__(self, w=640, h=480, block_size=20):
        # Inicia pygame
        pygame.init()
        
        # Dimensoes
        self.font = pygame.font.SysFont('arial', 25)

        self.w = w
        self.h = h
        self.block_size = block_size

        # Coordenadas do centro de distribuicao
        # self.cd_top = Point(self.w // 2, self.h * .1)
        # self.cd_left = Point(self.w * .4, self.h // 2)
        # self.cd_right = Point(self.w * .6, self.h // 2)
        # self.cd_bottom = Point(self.w // 2, self.h * .9)
        self.cd_center = Point(self.w/2, self.h/2)

        
        # Cria tela
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Drone')
        self.clock = pygame.time.Clock()


    def update(self, list_drones):
        # Preenche fundo
        self.display.fill(BLACK)
        # pygame.draw.circle(self.display, ORANGE, self.cd_top, 20) # cd norte
        # pygame.draw.circle(self.display, ORANGE, self.cd_left, 20) # cd leste
        # pygame.draw.circle(self.display, ORANGE, self.cd_right, 20) # cd oeste
        # pygame.draw.circle(self.display, ORANGE, self.cd_bottom, 20) # cd sul
        pygame.draw.circle(self.display, ORANGE, self.cd_center, 20) # cd centro

        for i in list_drones:
            drone_id = i.drone_id
            dest = i.dest
            drone = i.drone

            # rect_dest = pygame.Rect(self.dest.x, self.dest.y, BLOCK_SIZE, BLOCK_SIZE)
            # pygame.draw.rect(self.display, RED, rect_dest)
            pygame.draw.circle(self.display, BLUE, dest, 10) # Desenha Destino

            # Desenha Drone
            rect_drone = pygame.Rect(drone.x, drone.y, self.block_size, self.block_size)
            pygame.draw.rect(self.display, RED1, rect_drone)
            # pygame.draw.rect(self.display, RED2, pygame.Rect(rect_drone.x+4, rect_drone.y+4, 12, 12))
            texto_id = self.font.render(str(drone_id), True, WHITE)
            self.display.blit(texto_id, (drone.x+4, drone.y))

            # Desenha Linha
            pygame.draw.line(self.display, GREEN, rect_drone.center, dest, 2) 

            # text = self.font.render("Score: " + str(score), True, BLACK)
            # self.display.blit(text, [0, 0])

        # Atualiza a tela
        pygame.display.update()
        self.clock.tick(SPEED)

    def close(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()