import control as control

from logging import getLogger
logger = getLogger(__name__)


# Python Root
if __name__ == "__main__":
    meta_data = control.create_meta_data('./tiger_gray.stl')
    meta_data['save_snapshot'] = True

    control.run(meta_data)
