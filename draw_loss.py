import json
import numpy as np
import re
import matplotlib.pyplot as plt


def get_show_from_logger(logger_path):
    # with open(j_path, 'r') as load_f:
    #     logger_file = json.load(load_f)
    f = open(logger_path, encoding='utf-8')
    show_x_dict = {'kimg': []}
    show_y_dict = {'g_loss': [], 'd_loss': []}
    while True:
        line = f.readline()
        if line:
            iter_info = eval(line)
            for key, item in iter_info.items():
                if key in list(show_x_dict.keys()):
                    show_x_dict[key].append(item)
                if key in list(show_y_dict):
                    show_y_dict[key].append(item)
        else:
            break
    f.close()
    return show_x_dict, show_y_dict


def plt_show(x_c, y_c, x_name, y_name, title, **plot_kwargs):
    x_c, y_c = np.array(x_c), np.array(y_c)
    plt.plot(x_c, y_c, **plot_kwargs)
    if title =='g_loss':
        title = 'generator_loss'
    if title == 'd_loss':
        title = 'discriminator_loss'
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()
    plt.close()


def main(logger_j_path):
    show_x_dict, show_y_dict = get_show_from_logger(logger_j_path)
    for x_name, x_coordinates in show_x_dict.items():
        for y_name, y_coordinates in show_y_dict.items():
            plt_show(x_coordinates, y_coordinates, x_name, y_name, y_name)


if __name__ == '__main__':
    j_path = '/data3/lyz/cache/genforce/stylegan_mstar_28z_degree1-2/2021-4-17-19-14-54/log.json'
    main(j_path)
