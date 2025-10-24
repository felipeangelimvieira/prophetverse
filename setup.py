from setuptools import setup, find_packages

setup(
    name="prophetverse",
    version="0.10.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "sktime>=0.30.0",
        "numpyro>=0.19.0",
        "optax>=0.2.4",
        "graphviz>=0.20.3,<0.22.0",
        "scikit-base>=0.12.0",
        "skpro>=2.9.2,<3.0.0",
    ],
)
