import setuptools

version = {}
with open('pippin_naco/__init__.py') as fp:
    exec(fp.read(), version)

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pippin_naco',
    version=version['__version__'],
    author='Sam de Regt',
    author_email='regt@strw.leidenuniv.nl', 
    packages=['pippin_naco'],
    url='https://pippin-naco.readthedocs.io', 
    python_requires='>=3.10',
    license='GNU General Public License v3.0', 
    description='A comprehensive PDI pipeline for NACO data (PIPPIN)', 
    long_description=long_description, 
    install_requires=[
        'numpy>=1.21.4', 
        'matplotlib>=3.5.1', 
        'scipy>=1.7.3', 
        'astropy>=5.0', 
        'astroquery>=0.4.3', 
        'tqdm>=4.60.0', 
        'pathlib>=1.0.1', 
        ], 
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Science/Research', 
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10', 
        'Topic :: Scientific/Engineering :: Astronomy', 
    	], 
    include_package_data=True, 
    package_data={'pippin_naco': ['*.txt']}, 
    zip_safe=False,
    entry_points={'console_scripts': ['pippin_naco = pippin_naco.__main__:main']}
)
