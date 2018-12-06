from cffi import FFI
ffi = FFI()
ffi.cdef("int bisect_left(int arr[], int r, int x);")
import os
file_dir = os.path.abspath('/home/cwh/coding/TrackViz/util')
lib = ffi.verify("#include <bisect.c>", include_dirs=[file_dir], libraries=[])

if __name__ == '__main__':
    a = range(10)
    print lib.bisect_left(a, len(a)-1, 5)