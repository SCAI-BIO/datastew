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

setup(
    name='datastew',
    version='0.1.0',  # will be substituted in publish workflow
    packages=find_packages(),  # This will automatically find all packages and sub-packages
    url='https://github.com/SCAI-BIO/index',
    license='Apache-2.0 license',
    author='Tim Adams',
    author_email='tim.adams@scai.fraunhofer.de',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'matplotlib~=3.8.1',
        'numpy==1.25.2',
        'openai~=0.28.0',
        'openpyxl',
        'pandas==2.1.0',
        'pip==21.3.1',
        'plotly~=5.17.0',
        'python-dateutil==2.8.2',
        'python-dotenv~=1.0.0',
        'pytz==2023.3',
        'seaborn~=0.13.0',
        'sentence-transformers==2.3.1',
        'setuptools==60.2.0',
        'scikit-learn==1.3.2',
        'six==1.16.0',
        'thefuzz~=0.20.0',
        'tzdata==2023.3',
        'wheel==0.37.1',
        'aiofiles~=0.7.0',
        'python-multipart',
        'SQLAlchemy~=2.0.27',
        'scipy~=1.11.4',
        'pydantic~=1.10.14'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    include_package_data=True,  # Ensure non-Python files are included
)
