import tensorflow as tf
import numpy as np
import time
import gym


ENVIRONMENT = "CartPole-v0" # "LunarLander-v2"
RENDER_ENVIRONMENT = False
SEED = 1

N_TRAINING_EPISODES = 20
N_OBSERVATION_EPISODES = 10
MAX_EPISODE_DURATION = 120

SAVE_PATH = None #"./saved-networks/" + ENVIRONMENT + "/"


class Agent:
    """
    The policy gradient agent.
    """

    def __init__(self, state_shape, n_actions, discount_rate=0.99,
        learning_rate=0.01, save_path=None, load_path=None):

        # used to determine the shape of the network layers
        self.state_shape = state_shape
        self.n_actions = n_actions

        # save and load paths
        self.save_path = save_path
        self.load_path = load_path

        # episode stuff
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # training parameters
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

        # init the network, launch tensorflow session
        self.initialize_policy_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        #  load previously saved networks if any
        self.saver = tf.train.Saver()
        if self.load_path:
            checkpoint = tf.train.latest_checkpoint(
                checkpoint_dir=self.load_path)
            if checkpoint:
                print("Loading existing network at:", self.load_path)
                self.saver.restore(self.sess, checkpoint)

    def initialize_policy_network(self):

        # dummy code that I inserted so that tf.saver doesn't throw
        # an error before you've implemented the network. delete this
        # once you've done E1
        dummy = tf.get_variable("dummy", shape=[1,1])

        ### E1 START: INSERT CODE HERE ###
        ### E1 END ###

        ### E4 START: INSERT CODE HERE ###
        ### E4 END ###
        return
        
    def choose_action(self, s):
        ### E2 START: INSERT CODE HERE ###
        ### E2 END ###
        return 0

    def save_transition(self, s, a, r):
        self.episode_states.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)

    def train(self, episode_no):

        # convert actions to one-hot
        a_one_hot = np.zeros([len(self.episode_actions), self.n_actions])
        a_one_hot[np.arange(len(self.episode_actions)),
            list(map(int, self.episode_actions))] = 1.0

        # discount rewards
        r_discounted = np.zeros_like(self.episode_rewards)
        accum = 0.0
        for i in reversed(range(len(self.episode_rewards))):
            accum = accum * self.discount_rate + self.episode_rewards[i]
            r_discounted[i] = accum
        r_discounted -= np.mean(r_discounted)
        r_discounted /= np.std(r_discounted)

        ### E3 START: INSERT CODE HERE ###
        ### E3 END ###

        print("===================================================")
        print("Completed Episode:", episode_no)
        print("Episode Reward:", np.sum(self.episode_rewards))

        # reset the episode state, action, reward accumulators
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # put code to save network here
        if self.save_path:
            print("Saving to: " + self.save_path)
            self.saver.save(self.sess, self.save_path + "saved-network")
        print("===================================================")


"""
The code from here down runs the training loop; it's responsible for:
    (1) Intiailising the environment and agent;
    (2) Having the agent play the game according, choosing an action at
        each timestep according to the policy; and
    (3) Calling the training step at the end of each episode.
"""

# initialize environment
env = gym.make(ENVIRONMENT).unwrapped
env.seed(SEED)
n_actions = env.action_space.n
state_shape = env.observation_space.shape[0]

# initialize agent
agent = Agent(state_shape, n_actions, discount_rate=0.95, 
    learning_rate=0.02, save_path=SAVE_PATH, load_path=SAVE_PATH)

# run the training loop
max_reward = -250.0
for episode_no in range(N_TRAINING_EPISODES + N_OBSERVATION_EPISODES):
    
    # reset the game
    s = env.reset()
    done = False

    # run the episode
    start_t = time.clock()
    while not done:

        # render if switched on
        if episode_no > N_TRAINING_EPISODES: env.render()

        # agent chooses an action
        a = agent.choose_action(s)

        # execute action, cause state transition
        s_, r, done, _ = env.step(a)

        # save transition
        agent.save_transition(s, a, r)

        # compute reward accumulated so far
        episode_reward = np.sum(np.transpose(agent.episode_rewards))

        # if too much time has elapsed, then move on
        current_t = time.clock()
        if current_t - start_t > MAX_EPISODE_DURATION:
            done = True

        # transition to next state
        s = s_

    # train the agent
    agent.train(episode_no)

    # print out the max reward accumulated so far
    if episode_reward > max_reward:
        max_reward = episode_reward
    print("Max Reward:", max_reward)
    print("===================================================\n")
