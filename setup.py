from setuptools import setup, find_packages

setup(
    name='state_dict_to_h5',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "h5py >= 3.0",
        "numpy",
        "torch >= 2.0.0"
    ],
    entry_points={
        'console_scripts': [],
    },
    author='Johannes Klatt',
    author_email='cdrsonan@nova-vox.org',
    description='Python package for saving PyTorch model weights (as well as other nested data structures) to .h5 or .hdf5 files.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/CdrSonan/state_dict_to_h5',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
