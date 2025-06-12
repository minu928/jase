from pathlib import Path
from setuptools import setup, find_packages


version_dict = {}
with open(Path(__file__).parents[0] / "jmdx/_version.py") as this_v:
    exec(this_v.read(), version_dict)
version = version_dict["version"]
del version_dict


setup(
    name="jmdx",
    version=version,
    author="minu928",
    author_email="minu928@snu.ac.kr",
    url="https://github.com/minu928/jase",
    install_requies=["jax>=0.4.35", "numpy>=1.26.4"],
    description="Jax Molecular Dynamics eXtended",
    packages=find_packages(),
    keywords=["jax", "molecular dynamics", "dft"],
    python_requires=">=3.10",
    package_data={"": ["*"]},
    zip_safe=False,
)
