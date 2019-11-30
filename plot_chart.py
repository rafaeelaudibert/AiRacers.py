import fire
import matplotlib.pyplot as plt
import json


def main(files, legends, title: str = ''):
    files = files[1:-1].split(', ')  # Parse files correctly
    legends = legends[1:-1].split(', ')  # Parse legends correctly

    data_list = []
    for file in files:
        with open(file, 'r') as f:
            data_list.append(json.load(f))

    epochs = [len(data) for data in data_list]

    plots = [plt.plot(list(range(epochs[idx])), [data['fitness']
                                                 for data in all_data]) for idx, all_data in enumerate(data_list)]

    plt.ylabel('fitness')
    plt.xlabel('epochs')
    plt.legend((p[0] for p in plots), (legend for legend in legends))
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
