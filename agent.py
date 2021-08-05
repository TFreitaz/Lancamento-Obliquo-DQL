import numpy as np
import tensorflow.compat.v1 as tf


class Agent:
    # inicia parâmetros do objeto
    def __init__(
        self,
        sess: tf.Session,
        num_actions: int,
        num_states: int,
        gamma: float,
        min_experiences: int,
        max_experiences: int,
        batch_size: int,
    ):
        self.sess = sess  # sessão do Tensorflow (funcional apenas para TensorFlow v.1)

        self.num_actions = num_actions  # número de possíveis combinacoes para o par angulo-velocidade
        self.min_experiences = min_experiences
        self.max_experiences = max_experiences
        self.batch_size = batch_size

        # placeholders (?)
        self.states_ph = tf.placeholder(tf.float32, shape=(None, num_states))
        self.targets_ph = tf.placeholder(tf.float32, shape=(None,))
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))

        # experiências
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.next_states: list = []

        self.gamma = gamma  # taxa de importância de eventos futuros

        fc1 = tf.layers.dense(self.states_ph, 16, activation=tf.nn.relu)  # primeira camada da rede
        fc2 = tf.layers.dense(fc1, 32, activation=tf.nn.relu)  # segunda camada da rede
        fc3 = tf.layers.dense(fc2, 64, activation=tf.nn.relu)
        fc4 = tf.layers.dense(fc3, 32, activation=tf.nn.relu)
        self.Q_predicted = tf.layers.dense(fc2, self.num_actions, activation=None)  # camada de saída da rede
        vet_Q_predicted = self.Q_predicted[tf.one_hot(self.actions_ph, self.num_actions, on_value=True, off_value=False)]  # (?)

        loss = tf.losses.mean_squared_error(self.targets_ph, vet_Q_predicted)  # função de perda
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)  # otimizador

        self.init = tf.global_variables_initializer()  # inicialização das variáveis globais

    # toma uma ação para um dado estado
    def choose_action(self, state: int, eps: float):
        if np.random.random() < eps:
            return np.random.choice(range(self.num_actions))  # toma uma ação aleatória dentre as possibilidades
        return np.argmax(self.predict_one(state))  # toma uma ação através da DQL

    # insere experiências obtidas
    def add_experience(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def predict_one(self, state):
        states = np.atleast_2d(state)
        return self.sess.run(self.Q_predicted, feed_dict={self.states_ph: states})

    def train(self):
        if len(self.states) < self.min_experiences:
            return

        min_index = m if (m := len(self.states) - self.max_experiences) >= 0 else 0

        if self.batch_size:
            avaliables = range(min_index, len(self.states))
            indexes = np.random.choice(avaliables, size=self.batch_size, replace=False)
        else:
            indexes = range(min_index, len(self.states))

        selected_states = [self.states[i] for i in indexes]
        selected_actions = [self.actions[i] for i in indexes]
        selected_rewards = [self.rewards[i] for i in indexes]

        targets = selected_rewards  # valor incrementado pelas ações futuras. Não há ações futuras.

        feed_dict = {self.states_ph: selected_states, self.actions_ph: selected_actions, self.targets_ph: targets}
        self.sess.run(self.optimizer, feed_dict=feed_dict)
