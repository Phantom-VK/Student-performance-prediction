from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(path:str) -> List:
    """
    Will return a list of requirements / libraries from the path / file provided
    :param path: path to requirements file
    :return: List of requirements
    """
    requirements = []
    with open(path, 'r') as file:
        requirements = file.readline()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT  in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return  requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Vikramaditya Khupse",
    author_email="vikramadityakhupse@gmail.com",
    packages=find_packages(),
    install_requirements = get_requirements('requirements.txt')
)