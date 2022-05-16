# Group 26: MovieLens - Basic Recommender Systems and Extensions
## _A additional README, because we LOVE documentation_

[![N|Solid](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftownsquare.media%2Fsite%2F442%2Ffiles%2F2012%2F08%2Fbest-worst-summer-2012-movies.jpg%3Fw%3D1200%26h%3D0%26zc%3D1%26s%3D0%26a%3Dt%26q%3D89&f=1&nofb=1)](https://github.com/nyu-big-data/final-project-group-26)

The following details salient features of code and the process for execution:

## Overview
- Sourcing the shell script will create Hadoop aliases, Spark Session object, load modules on the cluster environment, and run automatic building of...
- Packaging modules on wheels and zipping them up cluster use automatically
- Run popularity baseline model selection from main routine 
- Run ALS model selection from main routine
- Write relevant csv files for post-processing on user HDFS

Let it be known that we hereby implement the requirements set by the legendary [Brian McFee](https://steinhardt.nyu.edu/people/brian-mcfee) on the provided project description README found in our [repo](https://github.com/nyu-big-data/final-project-group-26).

> In the final project, you will apply the tools you have learned in this class to solve a realistic,
> large-scale, applied problem. Specifically, you will build and evaluate a collaborative-filter based recommender system. Two extensions will be created for full credit. 

> In addition to all of your code, produce a final report (not to exceed 6 pages), describing your implementation, evaluation results, and extensions. Your report should clearly identify the contributions of each member of your group. If any additional software components were required in your project, your choices should be described and well motivated here.

## Additional Software Components

- [Anaconda 2020.11](https://www.anaconda.com/blog/individual-edition-2020-11) - Anaconda module providing needed libraries for running source code
- [Spark 3.0.1](https://spark.apache.org/docs/3.0.1/) - Large-scale data processing done on Apache Spark
- [NYU Peele Cluster](https://www.nyu.edu/life/information-technology/research-computing-services/high-performance-computing/high-performance-computing-nyu-it/hpc-supercomputer-clusters/peel-hadoop.html) - Peel is NYU's newest Hadoop cluster


And of course the code is found in our Github repository [here](https://github.com/nyu-big-data/final-project-group-26)!

## Background on Packaging for Nerds (Optional)

We use [setuptools](https://pypi.org/project/setuptools/) (extension for building wheels that provides the ```bdist_wheel``` setuptools command) in the project to package our custom classes and methods. For the sake of brevity here, the term package is used both as the name of a container of modules as well as the cabinet used to package it. Python’s first mainstream packaging format was the .egg file that was later replaced by the wheel (.whl), which contains all the files for a PEP 376 compatible install. Noting that *.whl file is similar to an *.egg in that it’s basically a *.zip file, we automatically scan for such file formats and rename the extension from *.whl to *.zip (see ```build.py```) just because it was discovered that the ```--py-files``` flag passes our resources to a temporary directory created on HDFS just for the lifetime of that application and, to avoid an import error, then load the files to the execution environment, like the JVM environment, using the `.config` properties in the main routine in a format (i.e. zip) that the PySpark API recognizes. To then get the dependency distribution to work with compiled extensions, first run 
```sh
python3 setup.py bdist_wheel
```
to ensure compatible binaries are included in your zip. This source distribution contains source code: this is not only Python code, but also the source code of any extension modules (usually in C or C++, but in Python here) bundled with the package. Source distributions contain a bundle of metadata files in a directory called "<package-name>.egg-info" that help with building and installing the package, but users don’t need to touch this. Note that we are also adding the minimal metadata necessary to create a package in ```setup.py``` that also looks for valid package contents. We also opt for the wheels distribution format as it lets you skip the build stage required with source distributions which thereby increases delivery and installation speed and avoids the requisite of having development tools and libraries installed on the system (i.e. user doesn't need to compile extension modules on their side if needed.) 

Since we are dealing with a server here, things get dicey so we must additionally unzip our archive on the destination node since Python will not import compiled extensions from zip files! Finally, the dependencies can now be deployed, unzipped, and included in the ```PYTHONPATH``` by including them in the terminal-side, which is put in shell script for convenience:
```
alias spark-submit='PYSPARK_PYTHON=$(which python) spark-submit \
 --py-files dist/Group26_MovieLens-0.5.0-py3-none-any.zip \
 main.py'
```

## First Time Setup
As a shortcut to this all (and avoiding the use of `requirements.txt` as we couldn't configure pip install on the Peele cluster) was to run the aforementioned all in one go by sourcing the shell script and then submitting the job. Quite intuitive, this was.

```sh
source shell_setup.sh
<distribution metadata generated here>
spark-submit
```

### Installing Distribution Packges on the Cluster: Packaging ```validated_models``` and ```dataset_split``` Modules 
We opted for a more modular approach that enforced more of a OOP approach. Modules provide a handy way to split the code into more files within a namespace; they are nothing but files containing Python code the main program can import if needed which then promotes maintainability and code re-usability. Packages are a handy way to group related modules altogether! Distribution packages we discussed prior then become versioned archive files that contain import packages, modules, and other resource files used for that project run on the cluster.

#### Popularity Baseline
What it do? How it work? Salient points of code?
```sh
SOME
CODE
HERE
```
#### Alternating Least Squares (ALS) 
What it do? How it work? Salient points of code?
```sh
SOME
CODE
HERE
```


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [wheel]: <https://wheel.readthedocs.io/en/stable/>
   [Python setup tools]: <https://grimoire.carcano.ch/blog/python-setup-tools/>
   [PyFiles on Spark]: <https://newbedev.com/i-can-t-seem-to-get-py-files-on-spark-to-work/>
