from setuptools import find_packages, setup


def load_requirements(filepath):
    with open(filepath) as fp:
        return [pkg_name.strip() for pkg_name in fp.readlines()]


setup(
    name='learnit',
    version='0.0.1',
    description='RIT Machine Learning Framework',
    url='',
    author='RIT',
    author_email='@recruit.ai',
    license='TBD',
    keywords='keywords',
    packages=find_packages(exclude=[]),
    install_requires=load_requirements("requirements.txt"),
    classifiers=[],
    )
