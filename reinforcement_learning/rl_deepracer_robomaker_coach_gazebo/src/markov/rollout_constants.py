from enum import Enum

from std_msgs.msg import ColorRGBA


class YamlKey(Enum):
    CAR_COLOR = "CAR_COLOR"
    BODY_SHELL_TYPE = "BODY_SHELL_TYPE"
    RACE_TYPE = "RACE_TYPE"


F1 = "f1"


class BodyShellType(Enum):
    DEFAULT = "deepracer"
    F1_2021 = "f1_2021"
    F1_CAR_11 = "f1_car_11"
    F1_CAR_12 = "f1_car_12"
    F1_CAR_13 = "f1_car_13"
    F1_CAR_14 = "f1_car_14"
    F1_CAR_15 = "f1_car_15"
    F1_CAR_16 = "f1_car_16"
    F1_CAR_17 = "f1_car_17"
    F1_CAR_18 = "f1_car_18"
    F1_CAR_19 = "f1_car_19"
    F1_CAR_20 = "f1_car_20"
    F1_CAR_21 = "f1_car_21"


DEFAULT_COLOR = ColorRGBA(0.0, 0.0, 0.0, 1.0)


class CarColorType(Enum):
    BLACK = "Black"
    GREY = "Grey"
    BLUE = "Blue"
    RED = "Red"
    ORANGE = "Orange"
    WHITE = "White"
    PURPLE = "Purple"
    F1_CAR_1E73FC = "f1_car_1e73fc"
    F1_CAR_7102BB = "f1_car_7102bb"
    F1_CAR_CF15FC = "f1_car_cf15fc"
    F1_CAR_EB0000 = "f1_car_eb0000"
    F1_CAR_FF9900 = "f1_car_ff9900"
    F1_CAR_FFFE01 = "f1_car_fffe01"
    F1_CAR_9EFC03 = "f1_car_9efc03"
    F1_CAR_1BD900 = "f1_car_1bd900"
    F1_CAR_73FDF9 = "f1_car_73fdf9"
    F1_CAR_000000 = "f1_car_000000"
    F1_CAR_BCCACC = "f1_car_bccacc"
    F1_CAR_FFFFFF = "f1_car_ffffff"


COLOR_MAP = {
    CarColorType.BLACK.value: ColorRGBA(0.1, 0.1, 0.1, 1.0),
    CarColorType.GREY.value: ColorRGBA(0.529, 0.584, 0.588, 1.0),
    CarColorType.BLUE.value: ColorRGBA(0.266, 0.372, 0.898, 1.0),
    CarColorType.RED.value: ColorRGBA(0.878, 0.101, 0.145, 1.0),
    CarColorType.ORANGE.value: ColorRGBA(1.0, 0.627, 0.039, 1.0),
    CarColorType.WHITE.value: ColorRGBA(1.0, 1.0, 1.0, 1.0),
    CarColorType.PURPLE.value: ColorRGBA(0.611, 0.164, 0.764, 1.0),
    CarColorType.F1_CAR_1E73FC.value: ColorRGBA(0.117, 0.450, 0.988, 1.0),
    CarColorType.F1_CAR_7102BB.value: ColorRGBA(0.443, 0.007, 0.733, 1.0),
    CarColorType.F1_CAR_CF15FC.value: ColorRGBA(0.811, 0.082, 0.988, 1.0),
    CarColorType.F1_CAR_EB0000.value: ColorRGBA(0.921, 0.000, 0.000, 1.0),
    CarColorType.F1_CAR_FF9900.value: ColorRGBA(1.000, 0.600, 0.000, 1.0),
    CarColorType.F1_CAR_FFFE01.value: ColorRGBA(1.000, 0.996, 0.003, 1.0),
    CarColorType.F1_CAR_9EFC03.value: ColorRGBA(0.619, 0.988, 0.011, 1.0),
    CarColorType.F1_CAR_1BD900.value: ColorRGBA(0.105, 0.850, 0.000, 1.0),
    CarColorType.F1_CAR_73FDF9.value: ColorRGBA(0.450, 0.992, 0.976, 1.0),
    CarColorType.F1_CAR_000000.value: ColorRGBA(0.000, 0.000, 0.000, 1.0),
    CarColorType.F1_CAR_BCCACC.value: ColorRGBA(0.737, 0.792, 0.800, 1.0),
    CarColorType.F1_CAR_FFFFFF.value: ColorRGBA(1.000, 1.000, 1.000, 1.0),
}
