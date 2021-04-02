import os
import re
import time
import json

from agent import Agent
from environment import Environment
from utils import plot_evolution, socket_send

import numpy as np
import tensorflow.compat.v1 as tf

from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Union

env_settings: Dict[str, Union[int, str]] = {
    "theta_disc": 91,  # número de pontos de discretização do ângulo de lancamento
    "vel_disc": 100,  # número de pontos de dicretização da velocidade de lancamento
    "max_dist": 50,  # máxima distancia possivel para o alvo
    "target_len": "random",  # comprimento do alvo, isto e, tolerância absoluta para sucesso
}

n_states = 2

n_episodes = 1000  # número de episodios a serem executados
decay = 0.021  # decaimento da taxa de aleatoriedade

agent_settings: Dict[str, int] = {
    "num_states": 2,  # número de parâmetros de um estado = (distancia)
    "gamma": 0,  # incremento por ações futuras
    "min_experiences": 0,  # mínimo de experiências aleatórias
    "max_experiences": 500,  # máximo de experiências aleatórias
    "batch_size": 0,  # tamanho do pacote aleatório a ser treinado em cada episódio
}

total_reward = 0  # recompensa total
min_eps = 0.01  # mínima taxa de aleatoriedade
max_eps = 1  # máxima taxa de aleatoriedade
verbose = 0  # tipo de output visível após a execução
pack_size = 5  # números de episódios considerados em cada média no plot resultante

data: Dict[str, Any] = {
    "agent_settings": agent_settings,
    "env_settings": env_settings,
    "training_settings": {"n_states": n_states, "n_episodes": n_episodes, "min_eps": min_eps, "max_eps": max_eps, "decay": decay},
}

env = Environment(**env_settings)  # ambiente configurado com parâmetros definidos

with tf.Session() as sess:
    # saver = tf.train.Saver()
    agent = Agent(sess, len(env.actions), **agent_settings)  # agente configurado com acoes definidas
    sess.run(agent.init)  # inicalização da sessão para o agente (?)

    all_episodes = range(n_episodes)  # index de episódios

    if verbose == 0:
        all_episodes = tqdm(all_episodes)  # barra de progressão por episódios

    start_training = time.time()

    for i in all_episodes:
        eps = min_eps + (max_eps - min_eps) * np.exp(-decay * i)  # cálculo da taxa de aleatoriedade

        state = env.reset()  # gera um novo ambiente = novo alvo
        action = agent.choose_action(state, eps)  # toma uma ação dado o ambiente

        if False:  # i % 10 == 0:
            try:
                theta, vel = env.actions[action]
                values = json.dumps((theta, vel, state[0], state[1]))
                socket_send(values)
            except Exception:
                pass

        reward, next_state = env.step(action, state)  # calcula os efeitos da ação tomada
        agent.add_experience(state, action, reward, next_state)  # absorve a experiência obtida
        agent.train()  # treino

        if verbose == 1 and i % 100 == 0:
            print("Reward:", reward)

    data["results"] = {"training_rewards": agent.rewards, "training_time": time.time() - start_training}

    print("\nLast 100 episode rewards average:", sum(agent.rewards[-100:]) / 100)
    print("Total reward:", sum(agent.rewards))
    print("Total training time:", data["results"]["training_time"], "s")

    #     socket_send('end')

    # saver.save(sess, 'data/test_session')

    plot_evolution(agent.rewards, pack_size, "Treino")

    print("\nTesting.\n")

    n_episodes = 100

    rewards = []

    for i in tqdm(range(n_episodes)):
        state = env.reset()  # gera um novo ambiente = novo alvo
        action = agent.choose_action(state, 0)  # toma uma ação dado o ambiente
        reward, next_state = env.step(action, state)  # calcula os efeitos da ação tomada
        rewards.append(reward)

    data["results"]["testing_rewards"] = rewards
    data["date"] = datetime.today().isoformat()

    if "data" not in os.listdir():
        os.mkdir("data")
    filename = re.sub(r"[\-\:\.]", "", data["date"])
    with open(f"data/{filename}.json", "w") as f:
        json.dump(data, f)

    plot_evolution(rewards, pack_size, "Teste")

    print("\nScore médio:", sum(rewards) / n_episodes)
