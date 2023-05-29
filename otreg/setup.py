from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'Transport-based shape registration'
LONG_DESCRIPTION = ''

setup(
    name='otreg',
    version= VERSION,
    author='Yikun Bai',
    author_email='yikun.bai@vanderbilt.edu',
    description= DESCRIPTION,
    long_description= LONG_DESCRIPTION,
    url='https://github.com/Baio0/sopt_shape',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
    install_requires=[
        'numpy>=1.20.0',
        'numba>=0.53.0',
        'torch>=1.9.0',
        'tqdm>=4.59.0',
        'scipy>=1.6.0',
    ],
)