# Debug levels
DEBUG_NONE = 0
DEBUG_BASIC = 1
DEBUG_VERBOSE = 2

_shared_data = {'CURRENT_LEVEL': DEBUG_NONE}


def set_level(level):
    _shared_data['CURRENT_LEVEL'] = level

def get_level():
    return _shared_data['CURRENT_LEVEL']