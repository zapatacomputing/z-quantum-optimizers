import time

from zquantum.core.gradients import finite_differences_gradient
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.functions import CallableWithGradient, FunctionWithGradient
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    optimization_result,
    construct_history_info,
)
from zquantum.core.typing import RecorderFactory
from scipy.optimize import OptimizeResult
from typing import Dict, Optional
import numpy
import numbers

def time_print(start,end,word: str = None):
    final = end - start
    minutes, seconds = divmod(final, 60)
    print("{} took {} min {} sec".format(word,int(minutes), round(seconds)))


class GDOptimizer(Optimizer):
    """
    Optimizer implementation to perform various SGD-like gradient-based optimization routines popular in
    machine learning, such as SGD with Momentum, ADAM, RMSprop, nesterov variants thereof, and other related algorithms.

    This code was adapted from:
       https://github.com/aspuru-guzik-group/tequila/blob/master/src/tequila/optimizers/optimizer_gd.py
    by Sumner Alperin-Lea, who is primary contributor to the code herein adapted as well.

    Attributes
    ---------
    method:
        a string, mainly used to select the function, f.
    f:
        a function for updating the suggested parameters of a cost_function object

    Methods
    -------
    minimize:
        perform a minimization using one of several SGD-like algorithms.
    stop_condition:
        determines whether or not optimization has been completed.
    take_step:
        perform a single optimization step on a compiled cost_function, starting from a given point.

    """

    def __init__(self,
                 method: str = 'sgd',
                 options: dict = None,
                 recorder: RecorderFactory = _recorder,
                 **kwargs):

        """
        Parameters
        ----------
        method: str: Default = 'sgd':
            string specifying which of the available methods to use for optimization. if not specified,
            then unmodified, stochastic gradient descent will be used.
        options: dict: Default = None
            a dictionary of options to be used during minimization.
            possible options are:
                lr:
                    a float. Hyperparameter: The learning rate (unscaled) to be used in each update;
                    in some literature, called a step size.
                beta:
                    a float. Hyperparameter: scales (perhaps nonlinearly) all first moment terms in any relavant method.
                rho:
                    a float. Hyperparameter: scales (perhaps nonlinearly) all second moment terms in any relavant method.
                    in some literature, may be referred to as 'beta_2'.
                epsilon:
                    a float. Hyperparameter: used to prevent division by zero in some methods.
                tol:
                    a float. If specified, minimize aborts when the difference in energies between two steps is smaller
                    than tol.
        recorder:
            a zquantum recorder object, which will be used to record information if histories are sought.
        kwargs
        """

        super().__init__(recorder=recorder)
        self.method = method.lower()
        method_dict = {
            'adam': self._adam,
            'adagrad': self._adagrad,
            'adamax': self._adamax,
            'nadam': self._nadam,
            'sgd': self._sgd,
            'momentum': self._momentum,
            'nesterov': self._nesterov,
            'rmsprop': self._rms,
            'rmsprop-nesterov': self._rms_nesterov}
        self.f=method_dict[self.method]

        if options is None:
            options = {}
        self.options = options

    def stop_condition(self, step, diff):
        if step==0:
            return False
        if self.tol is None:
            if self.maxiter is None:
                raise Exception('Cannot optimize without some condition for stopping; please specify tol or maxiter')
            else:
                return step >= self.maxiter
        else:
            if self.maxiter is None:
                return numpy.abs(diff)<self.tol
            else:
                return (numpy.abs(diff) < self.tol or step >= self.maxiter)

    def _minimize(
        self,
        cost_function: CallableWithGradient,
        initial_params: numpy.ndarray,
        keep_history: bool = False,
        **kwargs) -> OptimizeResult:


        """
        perform a gradient descent optimization of some cost function.
        Parameters
        ----------
        cost_function: Objective:
            the objective to optimize.
        initial_params: numpy.ndarray:
            initial parameters; the first step of optimization.
        maxiter: int, optional:
            Specify maximum number of iterations to perform.
        keep_history: bool:
            whether or not to store information as the optimzier progresses.


        kwargs
        Returns
        -------
        GDResults
            all the results of optimization.
        """

        if 'lr' not in self.options.keys():
            self.lr = 0.01
        else:
            self.lr = self.options['lr']
        if 'beta' not in self.options.keys():
            self.beta = 0.9
        else:
            self.beta = self.options['beta']
        if 'rho' not in self.options.keys():
            self.rho = 0.999
        else:
            self.rho = self.options['rho']
        if 'epsilon' not in self.options.keys():
            self.epsilon = 1.0 * 10 ** (-7)
        else:
            self.epsilon = self.options['epsilon']
        if 'tol' not in self.options.keys():
            self.tol = 1e-3
        else:
            self.tol = self.options['tol']

        if self.tol is not None:
            self.tol = abs(float(self.tol))

        assert all([k > .0 for k in [self.lr, self.beta, self.rho, self.epsilon]])

        if 'maxiter' in self.options.keys():
            self.maxiter=self.options['maxiter']
        else:
            self.maxiter=None

        if not hasattr(cost_function, "gradient"):
            cost_function = FunctionWithGradient(
                cost_function, finite_differences_gradient(cost_function)
            )

        if keep_history:
            cost_function = self.recorder(cost_function)

        gradients = cost_function.gradient

        initial_params=numpy.asarray(initial_params)
        step=0

        s=time.time()
        e = cost_function(initial_params)
        f = time.time()
        time_print(s,f,'first call')

        opt_value = e
        opt_params = initial_params
        moments = (numpy.zeros(len(initial_params)),numpy.zeros(len(initial_params)))

        last = e

        v, moments = self.f(step=step,
                            gradients=gradients,
                            moments=moments,
                            v=initial_params)
        step += 1
        s=time.time()
        e=cost_function(v)
        end=time.time()
        time_print(s,end,"step call {}".format(str(int(step))))
        if e < opt_value:
            opt_value = e
            opt_params = v

        diff = numpy.abs(e - last)
        last=e


        while not self.stop_condition(step,diff):
            v, moments = self.f(step=step,
                   gradients=gradients,
                   moments=moments,
                   v=v)
            step += 1
            s = time.time()
            e = cost_function(v)
            end = time.time()
            time_print(s, end, "step call {}".format(str(int(step))))
            if e < opt_value:
                opt_value = e
                opt_params = v

            diff = numpy.abs(e - last)
            last = e

        return optimization_result(
            opt_value=opt_value,
            opt_params=opt_params,
            nit=step,
            **construct_history_info(cost_function, keep_history)
        )

    def _adam(self, gradients, step,
              v, moments,
              **kwargs):
        t = step + 1
        s = moments[0]
        r = moments[1]
        start=time.time()
        grads = gradients(v)
        end=time.time()
        time_print(start,end,'adam gradient')
        start=time.time()
        s = self.beta * s + (1 - self.beta) * grads
        r = self.rho * r + (1 - self.rho) * numpy.square(grads)
        s_hat = s / (1 - self.beta ** t)
        r_hat = r / (1 - self.rho ** t)
        new = v - (self.lr * s_hat / (numpy.sqrt(r_hat) + self.epsilon))
        back_moments = [s, r]
        end=time.time()
        time_print(start,end,'adam update')

        return new, back_moments

    def _adagrad(self, gradients,
                 v, moments, **kwargs):
        r = moments[1]
        grads = gradients(v)

        r += numpy.square(grads)

        new = v - self.lr * grads / numpy.sqrt(r + self.epsilon)

        back_moments = [moments[0], r]
        return new, back_moments

    def _adamax(self, gradients,
                v, moments, **kwargs):

        s = moments[0]
        r = moments[1]
        grads = gradients(v)
        s = self.beta * s + (1 - self.beta) * grads
        r = self.rho * r + (1 - self.rho) * numpy.linalg.norm(grads, numpy.inf)
        new = v - self.lr * s/r
        back_moments = [s, r]
        return new, back_moments

    def _nadam(self, step, gradients,
               v, moments,
               **kwargs):

        s = moments[0]
        r = moments[1]
        t = step + 1
        grads = gradients(v)
        s = self.beta * s + (1 - self.beta) * grads
        r = self.rho * r + (1 - self.rho) * numpy.square(grads)
        s_hat = s / (1 - self.beta ** t)
        r_hat = r / (1 - self.rho ** t)
        rule = - self.lr * (self.beta * s_hat + (1 - self.beta) * grads / (1 - self.beta ** t)) / (
                        numpy.sqrt(r_hat) + self.epsilon)
        new = v + rule
        back_moments = [s, r]
        return new, back_moments

    def _sgd(self, gradients,
             v, moments, **kwargs):

        grads = gradients(v)
        new = v - self.lr * grads
        return new, moments

    def _momentum(self, gradients,
                  v, moments, **kwargs):

        m = moments[0]
        grads = gradients(v)

        m = self.beta * m - self.lr * grads
        new = v + m

        back_moments = [m, moments[1]]
        return new, back_moments

    def _nesterov(self, gradients,
                  v, moments, **kwargs):

        m = moments[0]
        interim = v + self.beta * m
        grads = gradients(interim)

        m = self.beta * m - self.lr * grads
        new = v + m

        back_moments = [m, moments[1]]
        return new, back_moments

    def _rms(self, gradients,
             v, moments,
             **kwargs):

        r = moments[1]
        s=time.time()
        grads = gradients(v)
        e=time.time()
        time_print(s,e,'rms gradients')
        s=time.time()
        r = self.rho * r + (1 - self.rho) * numpy.square(grads)
        new = v - self.lr * grads / numpy.sqrt(self.epsilon + r)
        e=time.time()
        time_print(s,e,'rms update')
        back_moments = [moments[0], r]
        return new, back_moments

    def _rms_nesterov(self, gradients,
                      v, moments,
                      **kwargs):

        m = moments[0]
        r = moments[1]


        interim = v + self.beta * m

        grads = gradients(interim)
        r = self.rho * r + (1 - self.rho) * numpy.square(grads)
        new = v + self.beta * m - self.lr * grads / numpy.sqrt(r)

        back_moments = [m, r]
        return new, back_moments