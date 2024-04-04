import seaborn as sns
import matplotlib.pyplot as plt



def generate_scatter_plot(X, Y, filename):
    sns.despine()
    sns.scatterplot(x=X, y=Y)
    plt.savefig(filename+".png", dpi=400)
    plt.close()

def generate_multiple_scatter_plots(X, Y1, Y2, filename):
    sns.despine()
    sns.scatterplot(x=X, y=Y1, label="ground_truth")
    sns.scatterplot(x=X, y=Y2, label="predicted")
    plt.savefig(filename+".png", dpi=400)
    plt.close()

def generate_line_plot(X, Y, filename):
    sns.despine()
    sns.lineplot(x=X, y=Y)
    plt.savefig(filename+".png", dpi=400)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.close()