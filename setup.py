from setuptools import setup, find_packages

#def requirements_from_file(file_name):
#    return open(file_name).read().splitlines()


setup(
    name='langchaintools',
    version='0.2.0',
    description="langchaintools",
    author='fuyu-quant',  
    packages=find_packages(),
    license='MIT'
)