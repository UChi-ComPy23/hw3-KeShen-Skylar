import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from warnings import warn


class ForwardEulerOutput(DenseOutput):

    def __init__(self, t_old, t, y_old, y):
        super(ForwardEulerOutput, self).__init__(t_old, t)
        self.y_old = y_old
        self.y = y

    def _call_impl(self, t):
        t = np.asarray(t)
        if t.ndim == 0:  # Single time point
            return self.y_old
        else:
            result = np.empty((self.y_old.shape[0], t.shape[0]))
            for i in range(t.shape[0]):
                result[:, i] = self.y_old
            return result


class ForwardEuler(scipy.integrate.OdeSolver):

    def __init__(self, fun, t0, y0, t_bound, vectorized=False,
                 support_complex=False, **extraneous):
        if extraneous:
            warn("Extraneous arguments passed to ForwardEuler: {}".format(extraneous))

        super(ForwardEuler, self).__init__(fun, t0, y0, t_bound, vectorized,
                                           support_complex)

        # Step size
        self.h = abs(extraneous.get('h', (t_bound - t0) / 100.0))
        self._t_old = None
        self._y_old = None

    def _step_impl(self):
        try:
            # Store previous state for dense output
            self._t_old = self.t
            self._y_old = self.y.copy()

            # Compute step
            f = self.fun(self.t, self.y)
            h = self.direction * self.h

            # Adjust step if exceeding the bound
            if self.direction * (self.t + h - self.t_bound) > 0:
                h = self.t_bound - self.t

            # Euler update
            self.t = self.t + h
            self.y = self._y_old + h * f

            return True, None

        except Exception as e:
            return False, f"Step failed: {str(e)}"

    def _dense_output_impl(self):
        """Return piecewise constant interpolation between t_old and t."""
        if self._t_old is None:
            t_old = self.t
            y_old = self.y
        else:
            t_old = self._t_old
            y_old = self._y_old

        return ForwardEulerOutput(t_old, self.t, y_old, self.y)