from setuptools import setup, find_packages

setup(
    name="entropy_estimators",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "scipy", "mpmath"],
    python_requires=">=3.7",
)

