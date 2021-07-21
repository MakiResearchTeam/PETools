import math
from numba import njit
from .low_pass_filter import LowPassFilter


@njit
def c__alpha(cutoff, freq):
    te = 1.0 / freq
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / te)


@njit
def c__cutoff(mincutoff, beta, edx):
    return mincutoff + beta * math.fabs(edx)


class OneEuroFilter(object):
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(c__alpha(self.__mincutoff, self.__freq))
        self.__dx = LowPassFilter(c__alpha(self.__dcutoff, self.__freq))
        self.__lasttime = None

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq  # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=c__alpha(self.__dcutoff, self.__freq))
        # ---- use it to update the cutoff frequency
        cutoff = c__cutoff(self.__mincutoff, self.__beta, edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=c__alpha(cutoff, self.__freq))

    def reset_values(self):
        self.__x.reset_values()
        self.__dx.reset_values()

