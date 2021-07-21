from numba import njit


@njit(fastmath=True)
def calc_filter(alpha, value, s):
    return alpha * value + (1.0 - alpha) * s


class LowPassFilter(object):

    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.reset_values()

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def reset_values(self):
        self.__y = self.__s = None

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = calc_filter(self.__alpha, value, self.__s)
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y
