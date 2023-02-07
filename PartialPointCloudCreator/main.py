import control as control

from logging import getLogger
logger = getLogger(__name__)


# Python Root
if __name__ == "__main__":
    meta_data = control.create_meta_data('./tiger_gray.stl') # , angle_degree=20, repeat=18 )
    meta_data['save_snapshot'] = True
    meta_data['save_format'] = 'xyz' # in ['xyz', 'bin', 'stl ascii', 'stl binary']

    control.run(meta_data)
