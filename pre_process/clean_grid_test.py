from util.file_helper import read_lines, write_line


def clean_grid():
    grid_test_path = 'top10/test_track.txt'
    new_test_path = 'top10/test_tracks.txt'
    grid_test_path = 'data/top10/test_track.txt'
    new_test_path = 'data/top10/test_tracks.txt'
    grid_test_lines = read_lines(grid_test_path)
    for i, grid_test_line in enumerate(grid_test_lines):
        tail = grid_test_line[4: -1]
        if i < 775:
            write_line(new_test_path, ('%04d' % (i + 251)) + tail)
        else:
            write_line(new_test_path, grid_test_line[0: -1])


if __name__ == '__main__':
    clean_grid()