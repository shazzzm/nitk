import setuptools

with open("README", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name='nitk',  
    version='0.1',
    #scripts=['nitk/*.py'] ,
    author="Tristan Millington",
    author_email="tristan.millington@gmail.com",
    description="A package for inferring sparse partial correlation networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shazzzm/nitk",
    packages=setuptools.find_packages(),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
    ],

 )