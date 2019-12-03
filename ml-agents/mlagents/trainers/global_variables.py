import os


def set_debug(value:bool):
    os.environ['debug'] = str(value)


def get_debug():
    return bool(os.environ.get('debug', 'False') == 'True')
