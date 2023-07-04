import numpy as np

from new_re_learner.agent import Agent
from plotter import plot_learning_curve
from re_learner import NetCoder
from simulation import Simulation
from utils import Clock


class PPOTrainer:
    modelFile = 'model/elevator_phill.dat'

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
        N = (24 * 60) / 8  # learn after N steps
        # N = 20  # learn after N steps
        agent = self.get_new_actor(sim)
        n_games = 300
        # n_games = 300

        figure_file = 'images/train_phill.png'

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

            dice_score = score / float(24 * 60)
            avg_waiting = sum(all_avg_waiting) / tact
            print(
                f'{i}. Episode\t|\tDay-Score: {score:.1f}\t|\tBest Avg. Day-Score: {best_score:.1f}\t|\t'
                f'Avg. Day-Score: {avg_score:.1f}\t|\t'
                f'Avg. Waiting Time: {avg_waiting:.2f}\t|\tRemaining: {remaining}/100')

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            # print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            #       'dice score %.1f' % (score / (24 * 60)),
            #       'Avg. Waiting Time %.2f' % avg_waiting,
            #       f'Avg. Remaining {remaining}/100',
            #       'run_steps', n_steps, 'learning_steps', learn_iters)
        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)
        x = [i + 1 for i in range(learn_iters)]
        plot_learning_curve(x, agent.loss_hist, None, 'Total Loss')
