from matplotlib import pyplot as plt


def draw_function(chf_funcs):
    for fn in chf_funcs:
        plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()
