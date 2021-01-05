import matplotlib.pyplot as plt


def plot_evolution(rewards: list, pack_size: int, game_name: str):
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
    plt.title(f"{game_name} DRL para lançamento oblíquo")
    plt.show()


def get_config(item: str, default, func=None, conf: dict = {}):
    if item in conf:
        if func:
            return func(conf[item])
        return conf[item]
    return default
