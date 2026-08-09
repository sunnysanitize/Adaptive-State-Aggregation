from enum import Enum

import numpy as np
import numpy.typing as npt

INDEX = np.int32
VALUE = np.float64

IndexArray = npt.NDArray[np.int32]
ValueArray = npt.NDArray[np.float64]


class Phase(Enum):

    GLOBAL = "global"
    AGGREGATE = "aggregate"
