# z-quantum-optimizers

## What is it?

`z-quantum-optimizers` is a module with basic optimizers to be used in workflows with [Orquestra](https://www.zapatacomputing.com/orquestra/) – a platform for performing computations on quantum computers developed by [Zapata Computing](https://www.zapatacomputing.com).

Currently this library includes the following optimizers:
- Grid search
- [Scipy optimizers](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [CMS-ES (Covariance Matrix Adaptation Evolution Strategy) ](https://github.com/CMA-ES/pycma)

It also supports the optimization of variational circuits.

## Usage

### Workflow
In order to use `z-quantum-optimizers` in your workflow, you need to add it as an `import` in your Orquestra workflow:

```yaml
imports:
- name: z-quantum-optimizers
  type: git
  parameters:
    repository: "git@github.com:zapatacomputing/z-quantum-optimizers.git"
    branch: "master"
```

and then add it in the `imports` argument of your `step`:

```yaml
- name: my-step
  config:
    runtime:
      language: python3
      imports: [z-quantum-optimizers]
```

Once that is done you can:
- use any `z-quantum-optimizers` function by specifying its name and path as follows:
```yaml
- name: optimize-circuit
  config:
    runtime:
      language: python3
      imports: [z-quantum-optimizers]
      parameters:
        file: z-quantum-optimizers/steps/optimizers.py
        function: optimize_variational_circuit
```
- use tasks which import `zquantum.optimizers` in the python code (see below).

### Python

Here's an example of how to use methods from `z-quantum-optimizers` in a python task:

```python
from zquantum.optimizers import ScipyOptimizer
optimizer = ScipyOptimizer(method='L-BFGS-B')
```

or use `optimizer-specs` parameter to make our code work with other backends too:

```python
from zquantum.core.utils import create_object
optimizer_specs = {{inputs.parameters.optimizer-specs}}
optimizer = create_object(optimizer_specs)
```

Even though it's intended to be used with Orquestra, `z-quantum-optimizers` can be also used as a standalone Python module.
This can be done by running `pip install .` from the `src/` directory.


## Development and contribution

You can find the development guidelines in the [`z-quantum-core` repository](https://github.com/zapatacomputing/z-quantum-core).

### Running tests

In order to run tests please run `pytest .` from the main directory.

In order for the tests related to the daemon optimizer to work you need to first specify FLASK_APP environmental variable:
`export FLASK_APP=/path/to/z-quantum-optimizer/src/python/orquestra/optimizers/daemon-optimizer/proxy/rest.py`.
