import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from file_helper import read_lines_and

camera_cnt = 6
train_track_path = 'market/training_track.txt'


def distribute_with_camera(persons_in_cameras):
    camera_distribution = [list() for i in range(camera_cnt)]

    def shuffle_person(img_name):
        if '.' not in img_name:
            return
        track_info = img_name.split('.')[0].split('_')
        person_id = track_info[0]
        for i in range(camera_cnt):
            if person_id in persons_in_cameras[i]:
                camera_distribution[i].append(
                    [float(track_info[1][1]), float(track_info[1][3]) + float(track_info[2]) / 100000])

    read_lines_and(train_track_path, shuffle_person)
    return camera_distribution


def count_person_in_camera(camera_num):
    persons = list()

    def count_person(img_name):
        person_id = img_name.split('.')[0].split('_')[0]
        if person_id not in persons:
            persons.append(person_id)

    read_lines_and('market/c%d_tracks.txt' % (camera_num + 1), count_person)
    # print(persons)
    return persons


def viz_data_for_market():
    persons_in_cameras = list()
    for i in range(camera_cnt):
        persons_in_cameras.append(count_person_in_camera(i))
    person_distribution4_camera = distribute_with_camera(persons_in_cameras)
    print(person_distribution4_camera)
    return person_distribution4_camera


def viz_camera(track_data, target_ax):
    # target_ax.scatter(track_data[0], track_data[1], s=10, c='g', marker='o')
    # plt.legend('x%d' % subplot_place)
    sns.kdeplot(track_data[0], track_data[1], shade=True, bw="silverman", ax=target_ax, cmap="Purples")


def viz_market():
    f, axes = plt.subplots(3, 2, figsize=(15, 10))
    for ax_s in axes:
        for ax in ax_s:
            ax.set_xlabel('Camera')
            ax.set_ylabel('Time')
    viz_data = viz_data_for_market()
    for i in range(camera_cnt):
        track_data = np.array(viz_data[i]).transpose()
        viz_camera(track_data, axes[i / 2][i % 2])
    plt.show()


def count_with_camera(persons_in_cameras):
    camera_distribution = [[[0 for i in range(camera_cnt)] for i in range(camera_cnt)] for i in range(camera_cnt)]

    def shuffle_person(img_name):
        if '.' not in img_name:
            return
        track_info = img_name.split('.')[0].split('_')
        person_id = track_info[0]
        for i in range(camera_cnt):
            if person_id in persons_in_cameras[i]:
                camera_distribution[i][int(track_info[1][1]) - 1][int(track_info[1][3]) - 1] += 1

    read_lines_and(train_track_path, shuffle_person)
    for i in range(camera_cnt):
        for j in range(camera_cnt):
            print(camera_distribution[i][j])
        print('=' * 40)
    return camera_distribution


def count_market():
    persons_in_cameras = list()
    for i in range(camera_cnt):
        persons_in_cameras.append(count_person_in_camera(i))
    camera_distribute = count_with_camera(persons_in_cameras)
    return camera_distribute


if __name__ == '__main__':
    viz_market()
