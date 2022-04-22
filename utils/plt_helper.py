from matplotlib import pyplot as plt


def draw_function(func, est):
    try:
        for i, s in enumerate(func):
            plt.step(est.event_times_, s, where="post", label=str(i))
        plt.ylabel("Survival probability")
        plt.xlabel("Time in days")
        plt.legend()
        plt.grid(True)
    except AttributeError:
        # raise err
        print('Не ОК')
