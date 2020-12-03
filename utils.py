import matplotlib.pyplot as plt


def plot_evolution(rewards: list, pack_size: int):
    to_plot = []
    mean = 0
    for i, reward in enumerate(rewards):
        if i % pack_size == 0:
            to_plot.append(mean / pack_size)
            mean = 0
        mean += reward
    plt.figure(figsize=(15, 4))
    plt.plot(range(len(to_plot)), to_plot)
    plt.axis((0, len(to_plot), -25, 50))
    plt.xlabel(f"Pack de {pack_size} episódios")
    plt.ylabel("Score")
    plt.title("Treino de DRL para lançamento oblíquo")
    plt.show()
