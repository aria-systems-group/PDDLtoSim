# Description

This repository contains the sourcr code 

1. Construct a two-player game abstraction from a PDDL file.
2. Synthesis a startegy using the `regret_syn_toolbox`  and simulate the strategy using pybullet.

## About

Work in progress

## Installing this project

* install [`anaconda`](https://www.anaconda.com/products/individual) or [`miniconda`](https://docs.conda.io/en/latest/miniconda.html)

* install [`spot`](https://spot.lrde.epita.fr/install.html) if you are going to construct a DFA using an LTL formula.

* clone this repo with:
 ```bash
git clone --recurse-submodules https://github.com/aria-systems-group/PDDLtoSim.git .
 ```

* change into this repo's directory:
 ```bash
cd PDDLtoSim
 ```
* create the `conda` environment for this library:
```bash
conda env create -f environment.yml
 ```

* activate the conda environment:
 ```bash
conda activate regret_syn_env
 ```

## Running the code

`cd` into the root directory, activate the conda `env`  and run the following command

```bash
python3 main.py
```

## Results

Here are some glimpses of the simulated strategy using this toolbox. In our simulation world we consider two region of interest. A human region and a robot region. We say that the human (not shown) can reach and manipulate the boxes placed on the right side (human region) but not the ones placed on the left (robot region).  

Robot building an arch with black boxes as supports and white box on top in either of the regions. The human intervenes twice and the robot is using a regret-minimizing strategy.

![](gifs/3_daig_reg_human.gif "arch building")


Robot placing objects in the specific pattern. The black box should be placed at the bottom location, the grey box in the middle and the white box should in the top location in either of the regions. 

![](gifs/arch_reg_human.gif "diagonal placement")

## Spot Troubleshooting notes

You can build `spot` from source, official git [repo](https://gitlab.lrde.epita.fr/spot/spot) or Debain package. If you do source intallation, then run the following command to verify your installation

```bash
ltl2tgba --version

```

If your shell reports that ltl2tgba is not found, add `$prefix/bin` to you $PATH environment variable by using the following command

```bash
export PATH=$PATH:/place/with/the/file

```

Spot installs five types of files, in different locations. $prefix refers to the directory that was selected using the --prefix option of configure (the default is /usr/local/).

1) command-line tools go into $prefix/bin/
2) shared or static libraries (depending on configure options)
   are installed into $prefix/lib/
3) Python bindings (if not disabled with --disable-python) typically
   go into a directory like $prefix/lib/pythonX.Y/site-packages/
   where X.Y is the version of Python found during configure.
4) man pages go into $prefix/man
5) header files go into $prefix/include

Please refere to the README file in the tar ball or on their github [page](https://gitlab.lrde.epita.fr/spot/spot/-/blob/next/README) for more datails on trouble shooting and installation.


## Contact

Please contact me if you have questions at :karan.muvvala@colorado.edu