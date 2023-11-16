from setuptools import find_packages, setup
import os
import re
from typing import List

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

def _parse_requirements(path: str) -> List[str]:
    """Returns content of given requirements file."""
    with open(os.path.join(path)) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]
        
VERSIONFILE = "jaxmarl/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
git_tar = f"https://github.com/FLAIROx/JaxMARL/archive/v{verstr}.tar.gz"

setup(
    name="jaxmarl",
    version=verstr,
    author="Foerster Lab for AI Research",
    description="Multi-Agent Reinforcement Learning with JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FLAIROx/JaxMARL",
    download_url=git_tar,
    keywords="MARL reinforcement-learning python jax",
    packages=find_packages(exclude=["baselines"]),
    python_requires=">=3.8",
    install_requires=_parse_requirements("requirements/requirements.txt"),
    extras_require={
        "dev": _parse_requirements("requirements/requirements-dev.txt"),
    },
    
)
