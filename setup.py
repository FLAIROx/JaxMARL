from setuptools import find_packages, setup
import os
from typing import List

def _parse_requirements(path: str) -> List[str]:
    """Returns content of given requirements file."""
    with open(os.path.join(path)) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]

setup(
    name="smax",
    version="0.0.1",
    author="FLAIR",
    description="Multi-Agent Reinforcement Learning with JAX",
    keywords="MARL reinforcement-learning python jax",
    packages=find_packages(exclude=["baselines"]),
    python_requires=">=3.8",
    install_requires=_parse_requirements("requirements/requirements.txt"),
    extras_require={
        "dev": _parse_requirements("requirements/requirements-dev.txt"),
    },
    
)
