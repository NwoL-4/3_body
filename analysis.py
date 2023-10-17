import matplotlib.pyplot as plt
from os import listdir, chdir, getcwd, path

from numpy import arange, load, ndarray, zeros, transpose
from pick import pick
from tqdm import tqdm


def load_file(file_name) -> tuple[ndarray, ndarray]:
    file = load(f'{file_name}')
    return file['arr_0'], file['arr_1']


def anim(coord, index_range):
    for i in tqdm(index_range):
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Визуализация задачи 3-ех тел с учетом запаздывания сигнала методом РК-4\n", fontsize=14)
        ax.plot(coord[:i + 1, 0, 0], coord[:i + 1, 1, 0], coord[:i + 1, 2, 0], color="#ff00ff")  # 1-я планета
        ax.plot(coord[:i + 1, 0, 1], coord[:i + 1, 1, 1], coord[:i + 1, 2, 1], color="red")  # 2-я планета
        ax.plot(coord[:i + 1, 0, 2], coord[:i + 1, 1, 2], coord[:i + 1, 2, 2], color="gold")  # 3-я планета
        ax.scatter(coord[i + 1][0, 0], coord[i + 1][1, 0], coord[i + 1][2, 0], color="#990099", marker="o", s=80,
                   label="1ая планета")
        ax.scatter(coord[i + 1][0, 1], coord[i + 1][1, 1], coord[i + 1][2, 1], color="darkred", marker="o", s=80,
                   label="2ая планета")
        ax.scatter(coord[i + 1][0, 2], coord[i + 1][1, 2], coord[i + 1][2, 2], color="goldenrod", marker="o", s=80,
                   label="3ая планета")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_zlabel("z", fontsize=14)
        ax.legend(loc="upper left", fontsize=14)
        plt.pause(0.000000001)


route: str = getcwd()

chdir(f'{route}')

folder = []

for name in listdir():
    if path.isdir(path.join(route, name)):
        folder.append(name)

title: str = 'Выберите загружаемую папку'
option, index = pick(folder, title, indicator='=>', default_index=0)
chdir(f'{route}\\{option}')

settings = load('Settings.npz')
print(f'Число итераций: {settings["NoI"]}\n'
      f'Временной шаг: {int(settings["dt"])}\n\n'
      f'Радиусы планет: {settings["radii"]}\n'
      f'Массы планет: {settings["masses"]}\n\n'
      f'                                 x     y     z\n'
      f'Начальные скорости 1 планеты: {transpose(settings["initial_speed"])[0]}\n'
      f'Начальные скорости 2 планеты: {transpose(settings["initial_speed"])[1]}\n'
      f'Начальные скорости 3 планеты: {transpose(settings["initial_speed"])[2]}\n'
      f'                                     x          y          z\n'
      f'Начальные координаты 1 планеты: {transpose(settings["initial_coordinate"])[0]}\n'
      f'Начальные координаты 2 планеты: {transpose(settings["initial_coordinate"])[1]}\n'
      f'Начальные координаты 3 планеты: {transpose(settings["initial_coordinate"])[2]}\n')


frame_step = int(settings["dt"])

slice_list = int(input('\nВведите периодичность вывода кадров (больше - быстрее отрисовывает) type=int '))
frame_steps = arange(0, (len(listdir()) - 1) * frame_step, frame_step)

list_coordinate = zeros((len(frame_steps + 1), 3, 3))

list_coordinate[0, :, :] = load_file(f'{-1}.npz')[0]

print('\n')
print('Загрузка данных....................................\n')
for frame in tqdm(range(len(frame_steps) - 1)):
    list_coordinate[frame + 1, :, :] = load_file(f'{frame_steps[frame]}.npz')[0]

print('Загрузка завершена....................................\n'
      'Идет визуализация....................................')

index_slice = arange(0, list_coordinate.shape[0], step=slice_list)

fig = plt.figure(figsize=(15, 10))
anim(coord=list_coordinate, index_range=index_slice)
plt.show()
