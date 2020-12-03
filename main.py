from agent import Agent
from environment import Environment

theta_disc = 91     # numero de pontos de discretizacao do angulo de lancamento
vel_disc = 100      # numero de pontos de dicretizacao da velocidade de lancamento
max_dist = 50       # maxima distancia possivel para o alvo
target_len = 4      # comprimento do alvo, isto e, tolerancia absoluta para sucesso

n_episodes = 100    # numero de episodios a serem executados

total_reward = 0    # recompensa total

env = Environment(theta_disc, vel_disc, max_dist, target_len)    # ambiente configurado com parametros definidos
agent = Agent(num_actions=len(env.actions))                      # agente configurado com acoes definidas

print('Iniciando simulação.')
for _ in range(n_episodes):
    state = env.reset()                                        # gera um novo ambiente = novo alvo
    action = agent.choose_action(state)                        # toma uma acao dado o ambiente
    reward, next_state = env.step(action, state)               # calcula os efeitos da acao tomada
    agent.add_experience(state, action, reward, next_state)    # absorve a experiencia obtida
    total_reward += reward                                     # incorpora recompensa no total
    
    print('Reward:', reward)
    
print('\nTotal reward:', total_reward)