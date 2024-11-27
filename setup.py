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
    version='0.4.0', # will be substituted in publish workflow
    packages=find_packages(),  # This will automatically find all packages and sub-packages
    url='https://github.com/SCAI-BIO/datastew',
    license='Apache-2.0 license',
    author='Tim Adams',
    author_email='tim.adams@scai.fraunhofer.de',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'matplotlib~=3.8.1',  # Updated version
        'numpy==1.25.2',  # Updated version
        'openai~=0.28.0',  # Updated version
        'openpyxl',
        'pandas==2.1.0',  # Updated version
        'pip==23.3',
        'plotly~=5.17.0',  # Updated version
        'python-dateutil==2.8.2',
        'python-dotenv~=1.0.0',
        'pytz==2023.3',
        'seaborn~=0.13.0',  # Updated version
        'sentence-transformers==2.3.1',
        'setuptools==60.2.0',  # Updated version
        'scikit-learn==1.3.2',  # Updated version
        'six==1.16.0',
        'thefuzz~=0.20.0',  # Updated version
        'tzdata==2023.3',
        'wheel==0.37.1',
        'aiofiles~=0.7.0',
        'python-multipart',
        'SQLAlchemy~=2.0.27',  # Updated version
        'scipy~=1.11.4',  # Updated version
        'pydantic~=2.5.0',
        'requests~=2.31.0',  # Added new requirement
        'uuid~=1.30',  # Added new requirement
        'weaviate-client~=4.9.0'  # Updated version
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
    python_requires='>=3.9',  # Specify minimum Python version
    keywords='data-harmonization LLM embeddings data-steward',
    project_urls={
        'Documentation': 'https://github.com/SCAI-BIO/datastew#readme',
        'Source': 'https://github.com/SCAI-BIO/datastew',
        'Tracker': 'https://github.com/SCAI-BIO/datastew/issues',
    },
)
