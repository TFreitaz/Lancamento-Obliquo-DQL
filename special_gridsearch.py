from ztools.ntf import telegram  # apagar - biblioteca particular

from agent import Agent
from environment import Environment
from utils import get_config

import os
import re
import sys
import time
import json
import random
import numpy as np
import tensorflow.compat.v1 as tf

from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any

param_grid = {
    "theta_disc": (5, 30),
    "vel_disc": (25, 100),
    "n_episodes": (10000, 22000),
    "min_experiences": (300, 1700),
    "max_experiences": (4500, 15000),
    "decay": (0.02, 0.15),
    #     'batch_size': (0, 1000),
    "max_dist": (10, 200),
    "min_eps": (0.04, 0.14),
    "max_eps": (0.5, 0.97),
}


def pick_value(a, b):
    return random.random() * (b - a) + a


def config_list(param_grid, k):
    return [{param: pick_value(*param_grid[param]) for param in param_grid} for _ in range(k)]


to_search = config_list(param_grid, int(sys.argv[1]))

for i, conf in enumerate(to_search):
    #     print(f'({i+1}/{len(to_search)})')

    tqdm.write(f"({i+1}/{len(to_search)})")
    env_settings = {
        "theta_disc": get_config("theta_disc", 6, int, conf),  # número de pontos de discretização do ângulo de lancamento
        "vel_disc": get_config("vel_disc", 10, int, conf),  # número de pontos de dicretização da velocidade de lancamento
        "max_dist": get_config("max_dist", 30, int, conf),  # máxima distancia possivel para o alvo
        "target_len": "random",  # comprimento do alvo, isto e, tolerância absoluta para sucesso
        "diameter": "random",
    }

    n_episodes = get_config("n_episodes", 14000, int, conf)  # número de episodios a serem executados
    decay = get_config("decay", 0.01, float, conf)  # decaimento da taxa de aleatoriedade

    agent_settings = {
        "num_states": 3,  # número de parâmetros de um estado = (distancia)
        "gamma": 0,  # incremento por ações futuras
        "min_experiences": get_config("min_experiences", 1000, int, conf),  # mínimo de experiências aleatórias
        "max_experiences": get_config("max_experiences", 12000, int, conf),  # máximo de experiências aleatórias
        "batch_size": get_config("batch_size", 0, int, conf),  # tamanho do pacote aleatório a ser treinado em cada episódio
    }

    total_reward = 0  # recompensa total
    min_eps = get_config("min_eps", 0.12, float, conf)  # mínima taxa de aleatoriedade
    max_eps = get_config("max_eps", 0.75, float, conf)  # máxima taxa de aleatoriedade
    verbose = 0  # tipo de output visível após a execução

    pack_size = 10  # números de episódios considerados em cada média no plot resultante

    data: Dict[str, Any] = {
        "agent_settings": agent_settings,
        "env_settings": env_settings,
        "training_settings": {"n_episodes": n_episodes, "min_eps": min_eps, "max_eps": max_eps, "decay": decay},
        "version": "3.1.2-otimizado",
    }

    env = Environment(**env_settings)  # ambiente configurado com parâmetros definidos
    print(f'Configuração: theta_disc={env_settings["theta_disc"]}  vel_disc={env_settings["vel_disc"]}')
    with tf.Session() as sess:
        # saver = tf.train.Saver()
        agent = Agent(sess, len(env.actions), **agent_settings)  # agente configurado com acoes definidas
        sess.run(agent.init)  # inicalização da sessão para o agente (?)

        all_episodes = range(n_episodes)  # index de episódios

        if verbose == 0:
            all_episodes = tqdm(all_episodes, unit="episodes")  # barra de progressão por episódios

        start_training = time.time()

        for i in all_episodes:
            eps = min_eps + (max_eps - min_eps) * np.exp(-decay * i)  # cálculo da taxa de aleatoriedade

            state = env.reset()  # gera um novo ambiente = novo alvo
            action = agent.choose_action(state, eps)  # toma uma ação dado o ambiente
            reward, next_state = env.step(action, state)  # calcula os efeitos da ação tomada
            agent.add_experience(state, action, reward, next_state)  # absorve a experiência obtida
            agent.train()  # treino

            if verbose == 1 and i % 100 == 0:
                print("Reward:", reward)

        data["results"] = {"training_rewards": agent.rewards, "training_time": time.time() - start_training}

        print("\nLast 100 episode rewards average:", sum(agent.rewards[-100:]) / 100)
        print("Total reward:", sum(agent.rewards))
        print("Total training time:", data["results"]["training_time"], "s")

        # saver.save(sess, 'data/test_session')

        # plot_evolution(agent.rewards, pack_size, 'Treino')

        print("\nTesting.\n")

        n_episodes = 1000

        rewards = []

        for i in tqdm(range(n_episodes), unit="episodes"):
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
            try:
                json.dump(data, f)
            except Exception as e:
                telegram("Erro no carregamento dos dados.\n\n" + str(e))  # apagar - biblioteca particular
                raise e

        # plot_evolution(rewards, p ack_size, 'Teste')

        print("\nScore médio:", sum(rewards) / n_episodes)

telegram("GridSearch completo.")  # - biblioteca particular
