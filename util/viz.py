import matplotlib.pyplot as plt
import seaborn as sns


def draw_line(y_s, x_s, y_label, x_label, title, line_color=None):
    plt.subplots()
    plt.plot(x_s, y_s, color=line_color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.show()
