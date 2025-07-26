from setuptools import setup, find_packages

setup(
    name="entropy_estimators",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "mpmath",
    ],
    author="Fermin Moscoso del Prado",
    author_email="fermosc@gmail.com",
    description="Entropy estimation library with multiple estimators including NSB",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)

