from setuptools import find_packages, setup

setup(
    name="bayesian_optimisation",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib",
        "gp @ git+https://github.com/cemlyn007/GaussianProcesses",
    ],
)
