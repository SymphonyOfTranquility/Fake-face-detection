import pickle

from matplotlib import pyplot as plt

DICT_KEY_TO_NAME = {
    'acc': 'acc',
    'val_acc': 'test acc',
    'loss': 'loss',
    'val_loss': 'test loss'
}


def plot_history(history, dict_keys, plt_id):
    plt.figure(plt_id, figsize=(16, 9))

    for key, label in dict_keys:
        plt.plot(history[key], label=label, linewidth=4)
        plt.scatter(len(history[key])-1, history[key][-1], s=400)

    plt.title(DICT_KEY_TO_NAME[dict_keys[0][0]])
    plt.xlabel('Epochs')
    plt.ylabel(dict_keys[0][0])
    plt.grid(True)
    plt.legend()


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 30})
    with open('train_history', 'rb') as f:
        history = pickle.load(f)
        keys = history.keys()
        for i, key in enumerate(keys):
            if key == 'lr' or 'val' in key:
                continue
            val_key = 'val_' + key
            plot_history(history, [(key, 'train'), (val_key, 'test')], i)
            print(key, min(history[key]), max(history[key]))
            print(val_key, min(history[val_key]), max(history[val_key]))

        plt.show()


