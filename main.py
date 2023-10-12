import time
from datetime import datetime
from os import path, makedirs, chdir, getcwd
from typing import Union

import matplotlib.pyplot as plt
from numba import prange, njit
from numpy import zeros, sum, array, arange, where, ndarray, abs, sqrt, append, savez, load, min, zeros_like, int64
from numpy.linalg import norm
from scipy.constants import G, c
from tqdm import tqdm


def rk4(time_step: float, mass_coord: ndarray, mass_speed: ndarray, function, time_rk4: float):
    k1, l1 = time_step * mass_speed, time_step * function(time_rk4)
    k2, l2 = time_step * (mass_speed + l1 / 2), time_step * function(time_rk4 + time_step / 2)
    k3, l3 = time_step * (mass_speed + l2 / 2), time_step * function(time_rk4 + time_step / 2)
    k4, l4 = time_step * (mass_speed + l3), time_step * function(time_rk4 + time_step)
    return mass_coord + (k1 + 2 * k2 + 2 * k3 + k4) / 6, mass_speed + (l1 + 2 * l2 + 2 * l3 + l4) / 6


def solve(time_solve: float) -> ndarray:
    force: ndarray = zeros((3, 3, 3))
    for number_i in prange(3):
        for number_j in prange(3):
            if number_j == number_i:
                continue
            force[:, number_i, number_j] = gravity_force(coord_first_part=coordinate[:, number_i],
                                                         index_second_part=number_j,
                                                         index_first_part=number_i, time_gravity=time_solve)
    result_force = sum(force, axis=2)

    return array([
        result_force[0] / masses,
        result_force[1] / masses,
        result_force[2] / masses
    ])


def calculate_delay_force(first_particle: ndarray, isp: int, time_delay: float) -> Union[int, ndarray]:
    tau: ndarray = array([-1e10, 0])

    value: int = 0
    change_time: ndarray = abs(timeline - time_delay - tau[-1])

    second_particle = 0
    while tau[-1] - tau[-2] > dt and value < 1e2 and time_delay >= timeline[where(change_time == min(change_time))][0]:
        if time_delay >= timeline[where(change_time == min(change_time))][0]:
            second_particle = coordinate[:, isp]
        else:
            second_particle = open_particle(isp, timeline[where(change_time == min(change_time))])
        delta_radius: ndarray = second_particle - first_particle
        module_delta_radius: float = norm(delta_radius)

        tau: ndarray = append(tau, module_delta_radius / c)

        value += 1

        change_time: ndarray = abs(timeline - time_delay - tau[-1])

    if time_delay == timeline[where(change_time == min(change_time))][0]:
        second_particle = coordinate[:, isp]
    elif time_delay > timeline[where(change_time == min(change_time))][0]:
        second_particle = open_particle(isp, timeline[where(change_time == min(change_time))])
    return second_particle


def gravity_force(index_first_part: int, coord_first_part: ndarray, index_second_part: int, time_gravity: float):
    particle = calculate_delay_force(coord_first_part, index_second_part, time_gravity)
    if type(particle) == int:
        return 0
    else:
        return G * masses[index_first_part] * masses[index_second_part] / (sum(
            (particle - coord_first_part) ** 2) ** (3 / 2)) * array(particle - coord_first_part)


@njit
def open_particle(number_particle, time_open):
    particle: ndarray = load(f'{time_open}.npz')['arr_0'][:, number_particle]
    return particle


def load_npz(file_name) -> array:
    particle = load(f'{file_name}')
    return particle['arr_0'], particle['arr_1']


def anim(coord):
    for i in tqdm(range(0, len(list_coordinate) - 2)):
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


if __name__ == '__main__':

    input_file = getcwd()

    with open('settings.txt', encoding='UTF-8') as file:
        settings = [line.rstrip() for line in file]

    current_time = datetime.now()
    time_now = str(current_time.strftime("%d-%m-%Y %H %M"))

    makedirs(path.join(input_file, time_now), exist_ok=True)

    chdir(f'{input_file}\\' + time_now)

    print('------------------------ ЗАДАЧА 3-ех ТЕЛ ------------------------\n')

    # Временной шаг
    dt = float(settings[0][settings[0].find('=') + 1:])
    # Число итераций
    NoI: int = int(settings[1][settings[1].find('=') + 1:])

    timeline: ndarray = arange(0, NoI, 1) * dt

    # Массы планет
    masses: ndarray = array([settings[i][settings[i].find('=') + 1:] for i in [4, 5, 6]], dtype=float)

    # Начальная скорость планет
    initial_speed: ndarray = zeros((3, 3))
    for index, row in enumerate([15, 16, 17]):
        iterator = settings[row][settings[row].find('=') + 1:]
        rate = [float(value) for value in iterator.split()]
        initial_speed[:, index] = rate

    # Радиусы планет
    radii: ndarray = array([settings[i][settings[i].find('=') + 1:] for i in [9, 10, 11]], dtype=float)

    # Начальные координаты планет
    initial_coordinate: ndarray = zeros((3, 3))
    for index, row in enumerate([21, 22, 23]):
        iterator = settings[row][settings[row].find('=') + 1:]
        rate = [float(value) for value in iterator.split()]
        initial_coordinate[:, index] = rate

    coordinate: ndarray = initial_coordinate
    speed: ndarray = initial_speed

    savez('Settings', NoI=NoI,
          masses=masses,
          initial_speed=initial_speed,
          radii=radii,
          initial_coordinate=initial_coordinate,
          dt=dt)

    savez(f'{int(-1)}', coordinate, speed)

    condition: bool = False

    collection = zeros((NoI, 3, 3))

    print('------------------------ Идет просчет ------------------------\n')

    for frame in tqdm(range(len(timeline))):
        if frame == range(len(timeline))[-1]:
            continue
        if not condition:
            coordinate, speed = rk4(time_step=dt,
                                    mass_coord=coordinate,
                                    mass_speed=speed,
                                    function=solve,
                                    time_rk4=frame * dt)

            collection[frame, :, :] = coordinate

            savez(f'{int(frame * dt)}', coordinate, speed)

            for i in range(3):
                for j in range(i + 1, 3):
                    if sum((coordinate[:, i] - coordinate[:, j]) ** 2) ** (1 / 2) <= radii[i] + radii[j]:
                        condition = True
                        print(f'Упс 😧😫, планеты {i + 1} и {j + 1} взорвались 😈😈😈😈😈😈😈😈😈😈\nМожете закрывать')
                        time.sleep(10)
                        exit()

    print('Расчет завершен')
    frame_step = dt

    slice_list = int(input('\nВведите периодичность вывода кадров (больше - быстрее отрисовывает) type=int '))
    frame_steps = arange(0, timeline[-1], step=frame_step, dtype=int64)[::slice_list]

    list_coordinate = zeros((len(frame_steps) + 1, 3, 3))

    list_speed = zeros_like(list_coordinate)
    coordinate, speed = load_npz(f'{-1}.npz')

    list_coordinate[0, :, :] = coordinate
    list_speed[0, :, :] = speed

    print('\n')

    for frame in tqdm(prange(len(frame_steps) - 1)):
        coordinate, speed = load_npz(f'{frame_steps[frame]}.npz')
        list_coordinate[frame + 1, :, :] = coordinate
        list_speed[frame + 1, :, :] = speed
    print('\nИдет визуализация....................................')

    fig = plt.figure(figsize=(15, 10))
    anim(list_coordinate)
    plt.show()
