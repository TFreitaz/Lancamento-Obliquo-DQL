import numpy as np


class Environment:
    # inicia parametros do objeto
    def __init__(self, theta_disc: int, vel_disc: int, max_dist: float, target_len: float):

        max_theta = 90  # angulo maximo de lançamento
        d_theta = max_theta / theta_disc  # discretizacao dos possiveis angulos

        max_vel = np.sqrt(max_dist * 9.81 / 2)  # maxima velocidade de lancamento possivel
        d_vel = max_vel / vel_disc  # discretizacao das possiveis velocidades

        self.success_reward = 50  # recompensa para um acerto

        self.max_dist = max_dist  # maxima distancia do alvo
        self.target_len = target_len  # comprimento do alvo (tolerancia absoluta)
        self.theta_range = np.arange(0, max_theta + d_theta, d_theta)  # lista de angulos discretos
        self.vel_range = np.arange(0, max_vel + d_vel, d_vel)  # lista de velocidades discretas

        self.actions = [
            (theta, vel) for theta in self.theta_range for vel in self.vel_range
        ]  # lista de acoes de par velocidade-angulo

    def step(self, action: int, state: tuple):

        theta, vel = self.actions[action]  # utiliza o index da acao selecionada para identificar angulo e velocidade
        d = 2 * vel ** 2 * np.sin(2 * theta) / 9.81  # calcula ponto final apos o lancamento com parametros selecionados

        dist = state  # distancia do alvo
        err = abs(dist - d)  # distancia entre o alvo e o ponto atingido (erro)

        reward = -err  # torna o erro uma recompensa negativa

        if err < self.target_len / 2:  # verifica se o ponto atingido esta dentro da tolerancia do alvo
            reward += self.success_reward  # insere recompensa por acerto

        next_state = err  # define onde o projetil parou como proximo estado

        return reward, next_state

    def reset(self):
        dist = np.random.random() * self.max_dist  # um ponto qualquer entre 0 e a distancia maxima
        return dist
