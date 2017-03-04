# -*- coding: UTF-8 -*-
import shutil
from xml.dom.minidom import parse
import xml.dom.minidom

from file_helper import write_line


def rename_3dpes():
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse("3dpes_train.al")
    collection = DOMTree.documentElement

    # 在集合中获取所有电影
    annotations = collection.getElementsByTagName("annotation")
    path_3dpes = '/Users/cwh/Mission/lab/video/data/3DPeS/RGB/'

    # 打印每部电影的详细信息
    for annotation in annotations:

        old_name = str(annotation.getElementsByTagName('name')[0].childNodes[0].data[11:])
        cid = int(annotation.getElementsByTagName('id')[0].childNodes[0].data)
        infos = old_name.split('_')
        pid = int(infos[0])
        print(old_name)
        frame = int(infos[3])
        new_name = '%04d_c%ds1_%d.bmp' % (pid, cid, frame)
        write_line('3dpes/c%d_tracks.txt' % cid, new_name)
        write_line('3dpes/training_track.txt', new_name)
        # shutil.copy(path_3dpes + old_name, path_3dpes + 'format/' + new_name)


if __name__ == '__main__':
    rename_3dpes()