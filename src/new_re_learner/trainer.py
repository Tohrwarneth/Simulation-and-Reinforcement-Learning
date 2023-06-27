import gym as gym
import numpy as np

from new_re_learner.agent import Agent
from plotter import plot_learning_curve
from re_learner import NetCoder
from simulation import Simulation
from utils import Clock


class PPOTrainer:
    modelFile = 'model/elevator_phill.dat'

    def train(self):
        sim = Simulation(show_gui=False, rl_decider=True)
        N = (20 * 60) / 8  # learn after N steps
        batch_size = 5
        n_epochs = 4
        alpha = 0.0003
        agent = Agent(n_actions=NetCoder.num_actions, batch_size=batch_size,
                      alpha=alpha, n_epochs=n_epochs,
                      input_dims=len(sim.get_game_state()))
        n_games = 300

        figure_file = 'plots/cartpole.png'

        # best_score = env.reward_range[0]
        best_score = 0
        score_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0

        for i in range(n_games):
            observation = sim.sim_reset()
            observation = NetCoder.normalize_game_state(observation)
            done = False
            score = 0
            tact = 0
            while not done:
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done = sim.run(action=action, step=True)  # step is run
                observation_ = NetCoder.normalize_game_state(observation_)
                n_steps += 1
                tact = Clock.tact
                score += reward
                agent.remember(observation, action, prob, val, reward, done)
                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'dice score %.1f' % (score/(24*60)),
                  'run_steps', n_steps, 'time_steps', tact, 'learning_steps', learn_iters)
        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)
