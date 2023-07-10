"""
Class created by phill of Neuralnet
modified from gym to Elevator Simulation
"""
import numpy as np

import utils
from new_re_learner.agent import Agent
from plotter import plot_learning_curve
from re_learner import NetCoder
from simulation import Simulation
from utils import Clock


class PPOTrainer:

    @classmethod
    def get_new_actor(cls, sim: Simulation):
        batch_size = 5
        n_epochs = 4
        alpha = 0.0003
        agent = Agent(n_actions=NetCoder.num_actions, batch_size=batch_size,
                      alpha=alpha, n_epochs=n_epochs,
                      input_dims=len(NetCoder.normalize_game_state(sim.get_game_state())))
        return agent

    def train(self):
        sim = Simulation(show_gui=False, rl_decider=True)
        N = (24 * 60) / 8  # learn 8 times per day
        agent = self.get_new_actor(sim)
        n_games = 320

        figure_file = 'images/train_phill.png'

        best_score = 0
        score_history = []

        learn_iters = 0
        n_steps = 0

        for i in range(n_games):
            observation = sim.sim_reset()
            observation = NetCoder.normalize_game_state(observation)
            done = False
            score = 0
            tact = 0
            all_avg_waiting = list()
            while not done:
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done = sim.run(action=action, step=True)  # step is run
                all_avg_waiting.append(observation_[1])
                remaining = observation_[2]
                observation_ = NetCoder.normalize_game_state(observation_)
                n_steps += 1
                tact = Clock.tact
                score += reward
                agent.remember(observation, action, prob, val, reward, done)
                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
                observation = observation_
            sim.shutdown()
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            avg_waiting = sum(all_avg_waiting) / tact
            print(
                f'{i}. Episode\t|\tDay-Score: {score:.1f}\t|\tBest Avg. Day-Score: {best_score:.1f}\t|\t'
                f'Avg. Day-Score: {avg_score:.1f}\t|\t'
                f'Avg. Waiting Time: {avg_waiting:.2f}\t|\tRemaining: {remaining}/{utils.Conf.totalAmountPerson}')

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            x = [i + 1 for i in range(len(score_history))]
            plot_learning_curve(x, score_history, figure_file)
