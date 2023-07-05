import numpy as np
import matplotlib.pyplot as plt

from utils import Logger, Conf


def plot_learning_curve(x, scores, figure_file, figure_name='Running average of previous 100 scores'):
    if isinstance(scores, list):
        running_avg = np.array(scores)
    else:
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title(figure_name)
    if figure_file is None:
        figure_file = 'images/'
        figure_file += figure_name.replace(' ', '_')
    plt.savefig(figure_file)


def draw_data(simulation) -> None:
    """
    Drawing data to plots
    :return: None
    """
    # gamma
    #
    fig, axs = plt.subplots(layout='constrained')
    plt.hist(simulation.personManager.scheduleTimes, bins=24 * 60, density=True)

    plt.xlabel('Zeit [Minuten]')
    plt.ylabel('Dichte')
    min_to_hour = lambda x: np.divide(x, 60)
    secax1 = axs.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
    secax1.set_xlabel('Zeit [Stunden]')

    plt.title('Gamma-Verteilung')

    Logger.log('Gamma-Verteilung')
    if Conf.showPlots:
        plt.show()

    # floors
    #
    floors = list()
    for f in simulation.personManager.homeFloors:
        floors.append(int(f) + 1)

    plt.hist(floors, bins=[i for i in range(1, Conf.maxFloor + 1)], density=True, alpha=0.8)

    plt.xlabel('Etagen')
    plt.ylabel('Dichte')

    plt.title('Stockwerk-Verteilung')

    Logger.log(plot_name='Etagen-Verteilung')
    if Conf.showPlots:
        plt.show()

    fig: plt.Figure = plt.figure()
    ax_motion: plt.Axes = fig.add_subplot(211)
    ax_avg: plt.Axes = fig.add_subplot(212)
    ax_motion.plot([i for i in range(24 * 60)], simulation.personManager.numberInMotion)
    ax_motion.set_xlabel('Zeit [Minuten]')
    ax_motion.set_ylabel('Personen')
    secax0 = ax_motion.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
    secax0.set_xlabel('Zeit [Stunden]')
    ax_motion.title.set_text('Reisende Personen')

    ax_avg.plot([i for i in range(24 * 60)], simulation.avgWaitingTime)
    ax_avg.set_xlabel('Zeit [Minuten]')
    secax1 = ax_avg.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
    secax1.set_xlabel('Zeit [Stunden]')
    ax_avg.set_ylabel('Wartezeit [Minuten]')
    fig.suptitle("Durchschnittliche Wartezeit")

    fig.tight_layout()

    Logger.log(plot_name='Durchschnittliche-Wartezeit')
    if Conf.showPlots:
        fig.show()

    # average waiting time
    #
    fig: plt.Figure = plt.figure()
    ax_avg: plt.Axes = fig.add_subplot(211)
    ax_final_avg: plt.Axes = fig.add_subplot(212)

    ax_avg.plot([i for i in range(24 * 60)], simulation.avgWaitingTime)
    ax_avg.set_xlabel('Zeit [Minuten]')
    secax1 = ax_avg.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
    secax1.set_xlabel('Zeit [Stunden]')
    ax_avg.set_ylabel('Wartezeit [Minuten]')
    fig.suptitle("Durchschnittliche Wartezeit")

    ax_final_avg.plot([i for i in range(24 * 60)], simulation.finalAvgWaitingTime)
    ax_final_avg.set_xlabel('Zeit [Minuten]')
    secax2 = ax_final_avg.secondary_xaxis('top', functions=(min_to_hour, min_to_hour))
    secax2.set_xlabel('Zeit [Stunden]')
    ax_final_avg.set_ylabel('Wartezeit [Minuten]')
    ax_final_avg.title.set_text('Finale Durchschnittliche Wartezeit')
    fig.tight_layout()
    if Conf.showPlots:
        fig.show()
