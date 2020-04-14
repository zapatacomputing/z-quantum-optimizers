# z-quantum-optimizers

## What is it?

`z-quantum-optimizers` is a module with basic optimizers to be used in workflows with [Orquestra](https://www.zapatacomputing.com/orquestra/) – a platform for performing computations on quantum computers developed by [Zapata Computing](https://www.zapatacomputing.com).

Currently this library includes the following optimizers:
- Grid search
- [Scipy optimizers](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [CMS-ES (Covariance Matrix Adaptation Evolution Strategy) ](https://github.com/CMA-ES/pycma)

It also includes a task templates for optimization of variational circuits.

## Usage

### Workflow
In order to use `z-quantum-optimizers` in your workflow, you need to add it as a resource:

```yaml
resources:
- name: z-quantum-optimizers
  type: git
  parameters:
    url: "git@github.com:zapatacomputing/z-quantum-optimizers.git"
    branch: "master"
```

and then import in a specific step:

```yaml
- - name: my-task
    template: template-1
    arguments:
      parameters:
      - param_1: 1
      - resources: [z-quantum-optimizers]
```

Once it's done you can:
- use any template from `templates/` directory
- use tasks which import `zquantum.optimizers` in the python code (see below).

### Python

Here's an example how to do use methods from `z-quantum-optimizers` in a python task:

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

Even though it's intended to be used with Orquestra, you can also use it as a standalone python module.
In order to install it run `pip install .` from the `src` directory.


## Development and contribution

You can find the development guidelines in the [`z-quantum-core` repository](https://github.com/zapatacomputing/z-quantum-core).

### Running tests

In order to run tests please run `pytest .` from the main directory.
