from setuptools import setup, find_packages

setup(
    name="res-mlp-tensorflow",
    packages=find_packages(exclude=["tests"]),
    version="0.0.1",
    license="MIT",
    description="Tensorflow Reimplementation of ResMLP",
    author="Ye Yint Htoon",
    author_email="yeyinthtoon.yyh@gmail.com",
    url="https://github.com/yeyinthtoon/tf2-resmlp",
    python_requires=">=3.6",
    install_requires=[
        "einops>=0.3",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.5"],
        "tensorflow with gpu": ["tensorflow-gpu>=2.5"],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
