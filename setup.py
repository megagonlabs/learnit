from setuptools import find_packages, setup


def load_requirements(filepath):
    with open(filepath) as fp:
        return [pkg_name.strip() for pkg_name in fp.readlines()]

setup(
    name='learnit',
    version='0.0.0rc1',
    description='A Tool for Machine Learning Beginners',
    url='https://github.com/megagonlabs/learnit',
    author='Megagon Labs',
    author_email='learnit@megagon.ai',
    license='Apache Software License (Apache License, Version 2.0)',
    keywords='machine learning, automated machine learning, AutoML, feature extraction',
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt"),    
    classifiers=[],
    python_requires='>=3.5',
)
