class CurveSmoother:
    """This class imitates the behavior of the TensorBoard smoothing slider by performing a running average.

    For instance, to smooth a loss curve, instantiate the class and pass the values to the smooth function one by one.

    :param smoothing_factor: controls the amount of smoothing. 0 = no smoothing, 1 = stuck at initial value.
    :raises ValueError: If the smoothing factor is outside the valid range [0, 1].
    """

    def __init__(self, smoothing_factor):
        if not 0 <= smoothing_factor <= 1:
            raise ValueError('Smoothing factor must lie between 0 and 1.')
        self._smoothing_factor = smoothing_factor
        self._last_smoothed_value = None

    def smooth(self, new_value):
        """Smooth the value based on all previous values passed to this function.

        :param new_value: The latest value of the curve that is to be smoothed.
        :return: smoothed value
        """
        if self._last_smoothed_value is None:
            smoothed_value = new_value
        else:
            smoothed_value = self._last_smoothed_value * self._smoothing_factor + (
                        1 - self._smoothing_factor) * new_value
        self._last_smoothed_value = smoothed_value
        return smoothed_value
