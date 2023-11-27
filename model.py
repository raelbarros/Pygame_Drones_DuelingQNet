import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Definicao da arquitetura da rede neural para Q-learning
class DuelingQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.to(DEVICE)
        x = self.feature_layer(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combina valor e vantagem para obter os Q-values
        advantage_mean = advantage.mean(dim=-1, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        return q_values


    def save(self, file_name='model.pth'):
        now = datetime.now()
        # Salva os parametros do modelo treinado
        model_folder_path = f'./data/{now.year}{now.month}{now.day}/'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


# Classe para treinar a rede neural Q-learning
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(DEVICE) # Move o modelo para a GPU, se disponível
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # Otimizador Adam
        self.criterion = nn.MSELoss() # Funcao de perda Mean Squared Error (MSE)

    def train_step(self, state, action, reward, next_state, done):
        # Converte as variaveis de entrada para tensores do PyTorch
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(action, dtype=torch.long).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float).to(DEVICE)

        # Lida com casos em que a entrada tem uma dimensao unica
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Q-values preditos com o estado atual
        pred = self.model(state)
    
        # Cria uma copia dos Q-values preditos para serem usados como alvo
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]

            # Atualiza Q-value com a recompensa futura descontada (gamma) apenas se o episodio nao estiver concluido
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Atualiza o Q-value correspondente a acao escolhida
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Calcula a perda (loss) entre os Q-values preditos e os Q-values alvo
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        # 3: Retropropagacao do erro e atualizacao dos pesos da rede
        loss.backward()
        self.optimizer.step()
