from agent import Agent
from environment import Environment
from utils import plot_evolution

import numpy as np
import tensorflow.compat.v1 as tf

from tqdm import tqdm


theta_disc = 91  # número de pontos de discretização do ângulo de lancamento
vel_disc = 100  # número de pontos de dicretização da velocidade de lancamento
max_dist = 50  # máxima distancia possivel para o alvo
target_len = 10  # comprimento do alvo, isto e, tolerância absoluta para sucesso

n_states = 1  # número de parâmetros de um estado = (distancia)

gamma = 0.98  # incremento por acoes futuras
n_episodes = 2000  # número de episodios a serem executados
min_eps = 0.01  # mínima taxa de aleatoriedade
max_eps = 1  # máxima taxa de aleatoriedade
decay = 0.002  # decaimento da taxa de aleatoriedade

total_reward = 0  # recompensa total

verbose = 0  # tipo de output visível após a execução
pack_size = 5  # números de episódios considerados em cada média no plot resultante

env = Environment(theta_disc, vel_disc, max_dist, target_len)  # ambiente configurado com parâmetros definidos

with tf.Session() as sess:
    agent = Agent(sess, len(env.actions), n_states, gamma)  # agente configurado com acoes definidas
    sess.run(agent.init)  # inicalização da sessão para o agente (?)

    all_episodes = range(n_episodes)  # index de episódios

    if verbose == 0:
        all_episodes = tqdm(all_episodes)  # barra de progressão por episódios

    for i in all_episodes:
        eps = min_eps + (max_eps - min_eps) * np.exp(-decay * i)  # cálculo da taxa de aleatoriedade

        state = env.reset()  # gera um novo ambiente = novo alvo
        action = agent.choose_action(state, eps)  # toma uma ação dado o ambiente
        reward, next_state = env.step(action, state)  # calcula os efeitos da ação tomada
        agent.add_experience(state, action, reward, next_state)  # absorve a experiência obtida
        agent.train()  # treino

        if verbose == 1 and i % 100 == 0:
            print("Reward:", reward)

    print("\nLast 100 episode rewards average:", sum(agent.rewards[-100:]) / 100)
    print("Total reward:", sum(agent.rewards))

plot_evolution(agent.rewards, pack_size)
