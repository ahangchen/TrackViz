from util.file_helper import read_lines


def get_shot_rate():
    shot_rate_s = read_lines('data/ep_en.txt')
    final_shot_rate = shot_rate_s[-1].split()
    return float(final_shot_rate[0]), float(final_shot_rate[1])