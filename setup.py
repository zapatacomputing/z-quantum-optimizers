import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="z-quantum-optimizers",
    version="0.2.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Core optimizers for Orquestra.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/z-quantum-optimizers ",
    packages=["zquantum.optimizers"],
    package_dir={"": "src/python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "z-quantum-core",
        "pytest>=5.3.5",
        "marshmallow>=3.4.0",
        "cma==2.7.0",
        "Werkzeug>=1.0.0",
        "flask>=1.1.2",
        "pyyaml==5.3.1",
    ],
)
