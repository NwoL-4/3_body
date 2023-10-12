import matplotlib.pyplot as plt
from os import listdir, chdir, getcwd, path

from numpy import arange, load, ndarray, zeros, zeros_like, transpose
from pick import pick
from tqdm import tqdm


def load_file(file_name) -> tuple[ndarray, ndarray]:
    file = load(f'{file_name}')
    return file['arr_0'], file['arr_1']


def anim(coord):
    for i in tqdm(range(0, len(list_coordinate) - 2), ):
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
route: str = 'C:\\Vsykoe\\Math.method\\3 тела'

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

frame_steps = arange(0, (len(listdir()) - 1) * frame_step, frame_step)[::slice_list]

list_coordinate = zeros((len(frame_steps + 2), 3, 3))

list_speed = zeros_like(list_coordinate)
coordinate, speed = load_file(f'{-1}.npz')

list_coordinate[0, :, :] = coordinate
list_speed[0, :, :] = speed

print('\n')

for frame in tqdm(range(len(frame_steps) - 1)):
    coordinate, speed = load_file(f'{frame_steps[frame]}.npz')
    list_coordinate[frame + 1, :, :] = coordinate
    list_speed[frame + 1, :, :] = speed

print('\nИдет визуализация....................................')

fig = plt.figure(figsize=(15, 10))
anim(list_coordinate)
plt.show()
