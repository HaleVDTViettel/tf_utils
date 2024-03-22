from setuptools import setup, find_packages

setup(
    name='tf-utils',
    version='0.0.1',
    url='https://github.com/HaleVDTViettel/tf-utils.git',
    author='HaleVDTViettel',
    author_email='thanhha.le2323@gmail.com',
    description='utilties for tensorflow 2.x.x',
    install_requires=['tensorflow >= 2.0.0'],
    packages=find_packages('tf_utils'),
    setup_requires=["wheel"],
    include_package_data=True,
)
