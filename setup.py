import io
import os
from setuptools import setup, find_packages

DESCRIPTION = 'Intelligent data steward toolbox using Large Language Model embeddings for automated Data-Harmonization.'

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
        return [line for line in lines if line and not line.startswith("#")]


requirements = parse_requirements(os.path.join(here, 'requirements.txt'))

setup(
    name='datastew',
    version='0.1.0',
    packages=find_packages(),  # This will automatically find all packages and sub-packages
    url='https://github.com/SCAI-BIO/index',
    license='Apache-2.0 license',
    author='Tim Adams',
    author_email='tim.adams@scai.fraunhofer.de',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords=["data stewardship", "Large language models"],
)
