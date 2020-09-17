import os
import glob
import shutil
import numpy as np
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from distutils.cmd import Command


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        install.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_FILES = ['./build', './dist', './__pycache__', './*.pyc', './*.tgz', './*.egg-info', './.eggs']
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(root_dir, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(root_dir):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, root_dir))
                print('Removing %s' % os.path.relpath(path))
                shutil.rmtree(path)


ext_modules = [
    Extension(
        'volkscv.utils.cocoapi.pycocotools._mask',
        sources=['volkscv/utils/cocoapi/common/maskApi.c',
                 'volkscv/utils/cocoapi/pycocotools/_mask.pyx'],
        include_dirs=[np.get_include(), 'volkscv/utils/cocoapi/common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]


setup(
    name='volkscv',
    version='0.1.0',
    packages=find_packages(),
    url='',
    license='',
    author='hxcai, chwang',
    author_email='chxlll@126.com wch.1993.2@live.com',
    description='A foundational python library for computer vision research and deployment projects',
    long_description=read('README.md'),
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0',
        'scipy',
        'numpy',
    ],
    ext_modules=ext_modules,
    python_requires='>3.6',
    cmdclass={
        'clean': CleanCommand,
    },
    zip_safe=False,
)
