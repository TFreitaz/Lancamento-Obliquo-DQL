import numpy as np

g = 9.81


class Environment:
    # inicia parametros do objeto
    def __init__(self, theta_disc: int, vel_disc: int, max_dist: float, target_len: float, diameter: float):

        max_theta = 90  # ângulo máximo de lançamento
        d_theta = max_theta / theta_disc  # discretização dos possíveis ângulos

        max_vel = np.sqrt(2 * max_dist * g)  # máxima velocidade de lançamento possível
        d_vel = max_vel / vel_disc  # discretização das possíveis velocidades

        self.success_reward = 150  # recompensa para um acerto

        self.max_dist = max_dist  # máxima distância do alvo
        self.target_len = target_len  # comprimento do alvo (tolerância absoluta)
        self.random_target_len = type(target_len) not in [float, int]  # definição de comprimento aleatório do alvo
        self.theta_range = np.arange(0, max_theta + d_theta, d_theta)  # lista de ângulos discretos
        self.vel_range = np.arange(0, max_vel + d_vel, d_vel)  # lista de velocidades discretas
        self.diameter = diameter
        self.random_diameter = type(diameter) not in [float, int]  # definição de comprimento aleatório do alvo

        self.actions = [
            (theta, vel) for theta in self.theta_range for vel in self.vel_range
        ]  # lista de ações de par velocidade-ângulo

    def step(self, action: int, state: tuple):

        theta, vel = self.actions[action]  # utiliza o index da ação selecionada para identificar ângulo e velocidade

        dist, target_len, diameter = state  # distância do alvo, comprimento do alvo e diâmetro do projétil

        d = (
            3 / 2 * vel ** 2 * np.sin(np.deg2rad(2 * theta))
            - (vel * np.cos(np.deg2rad(theta)) * np.sqrt((vel * np.sin(np.deg2rad(theta))) ** 2 + g * diameter))
        ) / g  # calcula ponto final apos o lancamento com parâmetros selecionados

        err = abs(dist - d)  # distância entre o alvo e o ponto atingido (erro)

        reward = -err / self.max_dist * 100  # torna o erro uma recompensa negativa

        if err + diameter / 2 < target_len / 2:  # verifica se o ponto atingido esta dentro da tolerância do alvo
            reward += self.success_reward  # insere recompensa por acerto

        next_state = err  # define o ponto onde o projétil parou como próximo estado

        return reward, next_state

    def reset(self):
        dist = np.random.random() * self.max_dist  # um ponto qualquer entre 0 e a distância máxima
        target_len = (np.random.random() * 0.12 + 0.05) * self.max_dist if self.random_target_len else self.target_len
        diameter = (np.random.random() * 0.7 + 0.05) * target_len if self.random_diameter else self.diameter
        return (dist, target_len, diameter)
