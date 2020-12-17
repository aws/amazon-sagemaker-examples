from enum import Enum
from std_msgs.msg import ColorRGBA


class YamlKey(Enum):
    CAR_COLOR = 'CAR_COLOR'
    BODY_SHELL_TYPE = 'BODY_SHELL_TYPE'
    RACE_TYPE = 'RACE_TYPE'


F1 = 'f1'


class BodyShellType(Enum):
    DEFAULT = 'deepracer'
    F1_2021 = 'f1_2021'


DEFAULT_COLOR = ColorRGBA(0.0, 0.0, 0.0, 1.0)


class CarColorType(Enum):
    BLACK = 'Black'
    GREY = 'Grey'
    BLUE = 'Blue'
    RED = 'Red'
    ORANGE = 'Orange'
    WHITE = 'White'
    PURPLE = 'Purple'


COLOR_MAP = {
    CarColorType.BLACK.value: ColorRGBA(.1, 0.1, 0.1, 1.0),
    CarColorType.GREY.value: ColorRGBA(0.529, 0.584, 0.588, 1.0),
    CarColorType.BLUE.value: ColorRGBA(0.266, 0.372, 0.898, 1.0),
    CarColorType.RED.value: ColorRGBA(0.878, 0.101, 0.145, 1.0),
    CarColorType.ORANGE.value: ColorRGBA(1.0, 0.627, 0.039, 1.0),
    CarColorType.WHITE.value: ColorRGBA(1.0, 1.0, 1.0, 1.0),
    CarColorType.PURPLE.value: ColorRGBA(0.611, 0.164, 0.764, 1.0)}
