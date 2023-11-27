
from model import DuelingQNet, QTrainer
from collections import namedtuple
from collections import deque
from display import Display
from helper import plot
from enum import Enum
import numpy as np
import random
import torch
import math
import sys


# Define a direcao possível do drone
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define uma tupla para representar um ponto no plano cartesiano
Point = namedtuple('Point', 'x, y')

NUM_DRONES = 2
BLOCK_SIZE = 20
MAX_MEMORY = 200_000
BATCH_SIZE = 1000
LR = 0.001

# Classe que representa o drone no ambiente
class DroneIA:

    def __init__(self, id=None, display=None):
        self.display = display

        self.n_games = 0 # Numero total de jogos jogados
        self.epsilon = max(5, 100 - 2 * self.n_games) # Taxa de exploracao
        self.gamma = 0.5 # Fator de desconto para recompensas futuras
        self.memory = deque(maxlen=MAX_MEMORY) # Armazena experiencias para treinamento
        self.model = DuelingQNet(12, 256, 3) # Modelo de rede neural
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # Treinador do modelo
        
        # Carrega um modelo pre-treinado
        self.model.load_state_dict(torch.load('./model/model_trained.pth'))
        self.model.eval()

        self.drone_id = id

        self.collision = 0
        self.plot_point_collision = []

        self.reset()


    def reset(self):
        # Configuracao inicial do drone
        
        # self.cd = random.choice([self.display.cd_left, self.display.cd_right])
        self.cd = self.display.cd_center
        self.direction = Direction.RIGHT
        self.drone = self.cd
        self.dest = None
        self.travel = 0
        self.frame_iteration = 0
        self.score = 0
        self.record = 0

        self._destination() # Define um novo destino para o drone


    def _destination(self):
        # Define um novo destino aleatório para o drone
        x = random.randint(0, (self.display.w-BLOCK_SIZE )//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.display.h-BLOCK_SIZE )//BLOCK_SIZE)*BLOCK_SIZE
        self.dest = Point(x, y)
        self.travel = 0


    def play_step(self, action, list_drones=None):
        self.frame_iteration += 1

        # 1.  Verifica se o jogo foi fechado
        self.display.close()
        
        # 2. Movimento do drone
        self._move(action, list_drones) 
        
        # 3. Verifica se houve colisao com outros drones ou com as bordas
        reward = 0
        game_over = False
        if self.is_collision(point=None, list_drones=list_drones) or self.frame_iteration > 10_000:
            game_over = True
            reward = -100
            return reward, game_over, self.score

        # 4. Verifica se o drone chegou ao destino
        if math.dist(self.drone, self.dest) < BLOCK_SIZE:
            self.travel +=1
            reward = 20
            self.dest = self.cd
            if self.travel > 1:
                self.score += 1
                reward += 10
                self._destination()

        # 5. Atualiza o display
        self.display.update(list_drones)

        # 6. Retorna o resultado do jogo e a pontuacao
        return reward, game_over, self.score


    def is_collision(self, point, list_drones, get_state=False):
        '''
        Verifica se há colisão entre o drone atual e outros drones.
        Retorna True se houver colisão, False caso contrário.
        '''
        if point is None:
            point = self.drone

        # Verica se e so a checagem de estado ou se realmente e um movimento valido
        if get_state:
            # Detecta colisao com as bordas
            # Dentro do método is_collision
            if point.x < 0 or point.x > self.display.w-BLOCK_SIZE  or point.y < 0 or point.y > self.display.h-BLOCK_SIZE:
                return True
            
            # Detecta colisao com outros drones
            for other_drone in list_drones:
                other_drone = other_drone.drone
                if other_drone != self.drone and math.dist(self.drone, other_drone) <= BLOCK_SIZE * 3:
                    return True  # Há uma colisão
                
        elif not get_state:
            # Detecta colisao com as bordas
            if point.x < 0 or point.x > self.display.w-BLOCK_SIZE  or point.y < 0 or point.y > self.display.h-BLOCK_SIZE:
                return True
            
            # Detecta colisao com outros drones
            for other_drone in list_drones:
                aux_drone = other_drone.drone
                if aux_drone != self.drone and math.dist(self.drone, aux_drone) <= BLOCK_SIZE:
                    self.collision +=1
                    self.plot_point_collision.append(self.drone)
                    return True  # Há uma colisão
                
        return False  # Não há colisão


    # Adicione o seguinte método para verificar e evitar colisões:
    def _find_safe_direction(self, list_drones):
        safe_directions = []

        for direction in [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]:
            next_point = self._get_next_position(direction)
            if not any(drone.is_collision(next_point, list_drones, True) for drone in list_drones if drone != self):
                safe_directions.append(direction)

        # Se nenhuma direção é segura, retorna a direção atual (sem movimento)
        return safe_directions[0] if safe_directions else self.direction


    def _move(self, action, list_drones):
        # Atualiza a direção e a posição do drone com base na ação escolhida

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Sem mudança na direção
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Virar para direita (direção: direita -> baixo -> esquerda -> cima)
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Virar para esquerda (direção: direita -> cima -> esquerda -> baixo)

        # Obtém a próxima posição após a rotação
        next_point = self._get_next_position(new_dir)

        # Evitar colisões
        if self.is_collision(next_point, list_drones, True):
            safe_direction = self._find_safe_direction(list_drones)
            next_point_safe = self._get_next_position(safe_direction)
            self.direction = safe_direction
            self.drone = next_point_safe
        else:
            self.direction = new_dir
            self.drone = next_point


    # Modifica a função _get_next_position para garantir que o drone permaneça próximo à sua posição atual:
    def _get_next_position(self, direction):
        x, y = self.drone.x, self.drone.y

        # Ajusta a próxima posição para garantir que o drone não se mova muito longe
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        return Point(x, y)


    def get_state(self, list_drones):
        '''
        Obtém o estado atual com base na posição do drone e possíveis colisões com outros drones em diferentes direções.
        Retorna um vetor de estado codificado.
        O estado consiste em informações sobre possíveis colisões à frente, à direita e à esquerda,
        direção de movimento do drone, localização do destino em relação ao drone.
        Cada elemento do vetor é binário, indicando a presença ou ausência de uma condição.
        '''

        point_l = self._get_next_position(Direction.LEFT)
        point_r = self._get_next_position(Direction.RIGHT)
        point_u = self._get_next_position(Direction.UP)
        point_d = self._get_next_position(Direction.DOWN)

        state = [
            # Verifica perigos nas proximidades
            (self.is_collision(point_r, list_drones, True)),  # direita
            (self.is_collision(point_l, list_drones, True)),  # esquerda
            (self.is_collision(point_u, list_drones, True)),  # cima
            (self.is_collision(point_d, list_drones, True)),  # baixo
            
            # direção do movimento
            self.direction == Direction.LEFT,
            self.direction == Direction.RIGHT,
            self.direction == Direction.UP,
            self.direction == Direction.DOWN,
            
            # localização do destino 
            self.dest.x < self.drone.x,  # dest left
            self.dest.x > self.drone.x,  # dest right
            self.dest.y < self.drone.y,  # dest up
            self.dest.y > self.drone.y  # dest down
        ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        # Adiciona a experiencia atual a memeria de replay do agente.
        # Experiencia consiste em um estado, acao tomada, recompensa recebida, novo estado e indicador de conclusao.
        self.memory.append((state, action, reward, next_state, done)) 


    def train_long_memory(self):
        # Treina o modelo usando uma amostra aleatoria da memoria de replay do agente.
        # A amostra e composta de estados, acoes, recompensas, novos estados e indicadores de conclusao.

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        # Treina o modelo com uma única experiência.
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Decide a acao a ser tomada com base no estado atual.
        self.epsilon -= self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: # Toma acao aleatoria
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


    def train(self, list_drones):
        plot_collision = []
        plot_scores = [] # Lista para armazenar as pontuacoes de cada jogo
        plot_mean_scores = [] # Lista para armazenar as medias das pontuacoes ao longo do tempo
        total_score = 0 # Variavel para rastrear a pontuacao total ao longo de varios jogos

        while True:
            # Pega estado atual do ambiente
            states_old = [drone.get_state(list_drones) for drone in list_drones]

            # Verifica os movimentos possíveis para cada agente
            final_moves = [drone.get_action(state_old) for drone, state_old in zip(list_drones, states_old)]

            # Executa o movimento no ambiente e obtenha o novo estado para cada agente
            rewards, dones, scores = zip(*[drone.play_step(final_move, list_drones) for drone, final_move in zip(list_drones, final_moves)])
            states_new = [drone.get_state(list_drones) for drone in list_drones]

            # Treina a memória de curto prazo para cada agente
            for i, drone in enumerate(list_drones):
                drone.train_short_memory(states_old[i], final_moves[i], rewards[i], states_new[i], dones[i])

            # Armazena a experiência na memória de longo prazo para cada agente
            for i, drone in enumerate(list_drones):
                drone.remember(states_old[i], final_moves[i], rewards[i], states_new[i], dones[i])

            # Treina a memória de longo prazo para cada agente e plota os resultados
            for i, drone in enumerate(list_drones):
                if dones[i]:
                    drone.n_games += 1
                    drone.train_long_memory()

                    if drone.score > drone.record:
                        drone.record = drone.score
                        drone.model.save()

                    print(f'Drone {drone.drone_id}, Jogo {drone.n_games}, Pontuação {drone.score}, Recorde: {drone.record}')

                    # Atualiza as listas para plotagem
                    plot_collision.append(self.collision)
                    plot_scores.append(drone.score)
                    total_score += drone.score
                    mean_score = total_score / drone.n_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores, plot_collision, self.plot_point_collision)

                    drone.reset()





if __name__ == '__main__':

    display = Display(block_size=BLOCK_SIZE)  # Cria uma única instância do ambiente de exibição

    num_drones = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_DRONES  # Obtém o número de drones da linha de comando ou usa 1 por padrão
    drones = [DroneIA(id=i+1, display=display) for i in range(num_drones)]  # Lista de drones
    # display.update(drones)

    for drone in drones:
        drone.train(drones)  # Inicia o treinamento para cada drone