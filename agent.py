import numpy as np
import tensorflow.compat.v1 as tf


class Agent:
    # inicia parametros do objeto
    def __init__(self, sess: tf.Session, num_actions: int, num_states: int, gamma: float):
        self.sess = sess  # sessão do Tensorflow (funcional apenas para TensorFlow v.1)

        self.num_actions = num_actions  # número de possíveis combinacoes para o par angulo-velocidade

        # placeholders (?)
        self.states_ph = tf.placeholder(tf.float32, shape=(None, num_states))
        self.targets_ph = tf.placeholder(tf.float32, shape=(None,))
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))

        # experiencias
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.next_states: list = []

        self.gamma = gamma  # taxa de importância de eventos futuros

        fc1 = tf.layers.dense(self.states_ph, 16, activation=tf.nn.relu)  # primeira camada da rede
        fc2 = tf.layers.dense(fc1, 32, activation=tf.nn.relu)  # segunda camada da rede
        self.Q_predicted = tf.layers.dense(fc2, self.num_actions, activation=None)  # camada de saída da rede
        vet_Q_predicted = self.Q_predicted[tf.one_hot(self.actions_ph, self.num_actions, on_value=True, off_value=False)]  # (?)

        loss = tf.losses.mean_squared_error(self.targets_ph, vet_Q_predicted)  # função de perda
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)  # otimizador

        self.init = tf.global_variables_initializer()  # inicialização das variáveis globais

    # toma uma acao para um dado estado
    def choose_action(self, state: int, eps: float):
        if np.random.random() < eps:
            return np.random.choice(range(self.num_actions))  # toma uma ação aleatória dentre as possibilidades
        return np.argmax(self.predict_one(state))  # toma uma ação através da DQL

    # insere experiencias obtidas
    def add_experience(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def predict_one(self, state):
        states = np.atleast_2d(state)
        return self.sess.run(self.Q_predicted, feed_dict={self.states_ph: states})

    def train(self):
        targets = self.rewards  # valor incrementado pelas acoes futuras. Nao ha acoes futuras.

        feed_dict = {self.states_ph: [[x] for x in self.states], self.actions_ph: self.actions, self.targets_ph: targets}
        self.sess.run(self.optimizer, feed_dict=feed_dict)
