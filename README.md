**Projeto Pygame_Drones_DuelingQNet**

---

**Descrição:**

Este projeto apresenta uma implementação de simulação de drones autônomos usando a biblioteca Pygame e uma Rede Neural do tipo Dueling Q-Network (DQN). Os drones são treinados para evitar colisões e atingir destinos específicos em um ambiente simulado.

---

**Instruções de Uso:**

1. **Instalação de Dependências:**
   Certifique-se de ter as dependencias instalada. Se não, você pode instalá-lo usando o seguinte comando:
    ```sh
    pip install -r requirements.txt 
    ```


2. **Execução do Projeto:**
Execute o arquivo `drone.py` para iniciar a simulação dos drones. Isso abrirá uma janela do Pygame, onde você pode observar o treinamento dos drones.

---

**Estrutura do Projeto:**

- `drone.py`: Arquivo principal que inicia a simulação.
- `display.py`: Classe responsável pela visualização da simulação usando a biblioteca Pygame.
- `helper.py`: Plota graficos com as metricas da simulação.
- `model.py`: Implementa a arquitetura da Rede Neural do tipo Dueling Q-Network.
- `model_old.py`: Implementa a arquitetura da Rede Neural do tipo Linear Q-Network.


---

**Configurações e Hiperparâmetros:**

- O arquivo `drone.py` contém configurações ajustáveis, como o número de drones, essa configuração também pode ser passada por argumento via linha de comando
    ```sh
    python drone.py <numero de drones> 
    ```

---

**Resultados e Aprendizados:**

- Os resultados do treinamento, incluindo gráficos e métricas, podem ser encontrados na pasta `data` durante a execução de cada rodada do jogo.


---

**Contribuições:**

Contribuições são bem-vindas!

---
