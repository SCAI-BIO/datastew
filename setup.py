import io
import os

from setuptools import setup

DESCRIPTION = 'Intelligent data steward toolbox using Large Language Model embeddings for automated Data-Harmonization.'

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name='intdex',
    version='0.0.2',
    packages=['index'],
    url='https://github.com/SCAI-BIO/index',
    license='Apache-2.0 license"',
    author='Tim Adams',
    author_email='tim.adams@scai.fraunhofer.de',
    description=DESCRIPTION,
    long_description=DESCRIPTION,
)
