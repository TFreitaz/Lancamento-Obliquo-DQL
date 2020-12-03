import numpy as np


class Agent:
    # inicia parametros do objeto
    def __init__(self, num_actions: int):
        self.num_actions = num_actions  # numero de possiveis combinacoes para o par angulo-velocidade

        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.next_states: list = []

    # toma uma acao para um dado estado
    def choose_action(self, state):
        return np.random.choice(range(self.num_actions))  # toma uma acao aleatoria dentre as possibilidades

    # insere experiencias obtidas
    def add_experience(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)