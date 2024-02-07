from setuptools import setup
from pathlib import Path

requirements_file = Path("requirements.txt")
requirements_list = []
if requirements_file.is_file():
    with open("requirements.txt", encoding="utf-8") as requs:
        requirements_list = [requ.strip("\n \r") for requ in requs.readlines()]

setup(
    name="tttg_fork_base_python",
    version="0.1",
    description="Base repo to fork when starting new python project",
    url="https://github.com/TomTomToGo-Github/tttg_fork_base_python.git",
    author="Thomas Haid",
    author_email="thomas.haid@gmx.net",
    install_requires=requirements_list,
    license="MIT",
    packages=[],
    # zip_safe=
)
