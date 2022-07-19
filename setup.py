from setuptools import setup

setup(
    name='pippin',
    version='0.0.1',
    author='S. de Regt',
    packages=['pippin'],
    install_requires=['numpy >= 1.21.4',
                      'matplotlib >= 3.5.1',
                      'scipy >= 1.7.3',
                      'astropy >= 5.0',
                      'astroquery >= 0.4.3',
                      'tqdm >= 4.60.0',
                      'pathlib >= 1.0.1',
                      ],
    python_requires='>=3',
    entry_points={'console_scripts': ['pippin = pippin.__main__:main']}
)
