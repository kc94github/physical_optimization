from setuptools import setup, find_packages

setup(
    name="drivepathsolver",
    version="0.1.2",
    author="Kailiang Chen",
    author_email="kailiangchen94@gmail.com",
    description="Use QP solver to optimize paths for robots",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kc94github/physical_optimization/tree/dev/python38",  # Optional project URL
    packages=find_packages(),
    install_requires=[
        "qpsolvers==4.0.0",
        "numpy==1.24.4",
    ],
    classifiers=[
        # Choose your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    # Additional metadata about your package
    keywords="robotics, path, optimization, qp",
)
