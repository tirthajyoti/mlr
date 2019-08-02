from io import open
from setuptools import setup, Extension

def read(fname, encoding='utf-8'):
    with open(fname, encoding=encoding) as f:
        return f.read()

setup(
    name='mlr',
    version="0.1.0",
    author='Tirthajyoti Sarkar',
    author_email='tirthajyoti@gmail.com',
    description='Linear regression utility with inference tests, residual analysis, outlier visualization, multicollinearity test, and other features',
    url='https://github.com/tirthajyoti/mlr',
    license='GPLv3+',
    long_description_content_type='text/markdown',
    long_description=read('README.md'),
    packages=['mlr'],
    install_requires=['numpy','pandas','matplotlib','seaborn','statsmodels'],
    keywords=[
        'Regression',
        'Linear regression',
        'Data science',
        'Machine learning',
        'Engineering',
        'Statistics',
        'Modeling',
        'Analytics',
        'Predictive analytics',
        'Data mining'
        ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
        ]
    )
