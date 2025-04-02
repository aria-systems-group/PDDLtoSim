# Description

This repository contains code for the paper:

- Let's Collaborate: Regret-based Reactive Synthesis for Robotic Manipulation (ICRA 22) - ([branch](https://github.com/aria-systems-group/PDDLtoSim))
- Efficient Symbolic Approaches for Quantitative Reactive Synthesis with Finite Tasks (IROS 23) - ([branch](https://github.com/aria-systems-group/PDDLtoSim/tree/experiments/iros_sims_fixes))

Primarily, we use this code to
1. Parse the PDDL File, construct a two-player turn-based game abstraction from a PDDL file. The syntax of the game is same as [FOND](https://link.springer.com/chapter/10.1007/978-3-031-01564-9_1). 
2. Synthesize a strategy using this [toolbox](https://github.com/aria-systems-group/regret_synthesis_toolbox/tree/master), and
3. Simulate the strategy using [Pybullet](https://pybullet.org/wordpress/).

**Note**: the Pybullet sim is old will be phased out soon with probably a better simulator. Further, the keyword `:non-deterministic` and the additional operator `oneof` is not yet supported. These are standard ways of specifying non-deterministic outcomes under an action using the PDDL Semantics (Refer [PRP Paper](https://ojs.aaai.org/index.php/ICAPS/article/view/13520), [PRP Code](https://github.com/QuMuLab/planner-for-relevant-policies)). In my file, I add non-determinism as `human-move` action (See [my code](https://github.com/aria-systems-group/sym_quant_reactive_synth/blob/master/pddl_files/dynamic_franka_world/domain.pddl)).


# Authors

* [Karan Muvvala](https://muvvalakaran.github.io/) <[karan.muvvala@colorado.edu](mailto:karan.muvvala@colorado.edu)>


# Installation

## Clone the code

* clone this repo with:
 ```bash
git clone --recurse-submodules https://github.com/aria-systems-group/PDDLtoSim.git .
 ```

My Repositiory depends on multiple submodules and each branch possibly has different submodule version in it.  The `--recurse-submodule` will automatically initialize and update each submodule in the repository. However, this command will **only** update all submodules (including nested ones) to the commit that's referenced by **the current branch** of the parent repository. If you want to automate this process, use the `--recurse-submodules=on-demand` option with `git checkout`, like this:

```bash
git checkout --recurse-submodules=on-demand branch-name
```

This will automatically update the submodules when switching branches.

## Docker Installation - Creating an Image and Spinning a Container

Make sure you have Docker installed. If not, follow the instructions [here](https://docs.docker.com/engine/install).

### Docker Commands to build the image

1. `cd` into the root of the project

2. Build the image from the Dockerfile

```bash
docker build -t <image_name> .
```

Note: the dot looks for a Dockerfile in the current repository. Then spin an instance of the container by using the following command

```bash
docker run -it --name <docker_container_name> <docker image name>
```

For volume binding

```bash
docker run -v <HOST-PATH>:<Container-path>
```

For example, to volume bind your local directory to the `pddltosim` folder inside the Docker, use the following command

```bash
docker run -it -v $PWD:/root/pddltosim --name <dokcer_container_name> <image_name>
```

Here `<docker_container_name>` is any name of your choice and `<image_name>` is the docker image name from above. `-it` and `-v` are flags to run an interactive terminal and volume bind respectively. 

Additionally, if you are more used to GUI and would like to edit or attach a container instance to VSCode ([Link](https://code.visualstudio.com/docs/devcontainers/containers)) then follow the instructions below:

### Attaching the remote container to VScode


1. Make sure you have the right VS code extensions installed
   * install docker extension
   * install python extension
   * install remote container extension
   * Now click on the `Remote Explore` tab on the left and attach VScode to a container.
2. This will launch a new vs code attached to the container and prompt you to a location to attach to. The default is root, and you can just press enter. Congrats, you have attached your container to VSCode.


## Conda Installation - Instructions to create the env for the code

* install [`anaconda`](https://www.anaconda.com/products/individual) or [`miniconda`](https://docs.conda.io/en/latest/miniconda.html)

* install [`spot`](https://spot.lrde.epita.fr/install.html) if you are going to construct a DFA using an LTL formula.


* change into this repo's directory:
 ```bash
cd PDDLtoSim
 ```
* create the `conda` environment for this library:
```bash
cd conda && conda env create -f environment.yml
 ```

* activate the conda environment:
 ```bash
conda activate regret_syn_env
 ```

### Running the code

`cd` into the root directory, activate the conda `env`  and run the following command

```bash
python3 main.py
```

## Tests

All the tests related scripts are available in the `tests/` directory. I use python [unittest](https://docs.python.org/3.8/library/unittest.html) for testing individual components of my source code. Here are some commands to run the tests:

To run a specific test package:

```bash
python3 -m unittest discover -s tests.<directory-name> -bv
```

To run a specific test script:

```bash
python3 -m tests.<directory-name>.<module-nane> -b
```

To run all tests:

```bash
python3 -m unittest -bv
```

For more details see the `tests/README.md`. Note, all commands must be run from `<root/of/project>`.

## Results

Here are some glimpses of the simulated strategy using this toolbox. In our simulation world we consider two region of interest. A human region and a robot region. We say that the human (not shown) can reach and manipulate the boxes placed on the right side (human region) but not the ones placed on the left (robot region).  

Robot building an arch with black boxes as supports and white box on top in either of the regions. The human intervenes twice and the robot is using a regret-minimizing strategy.

![](gifs/3_daig_reg_human.gif "arch building")


Robot placing objects in the specific pattern. The black box should be placed at the bottom location, the grey box in the middle and the white box should in the top location in either of the regions. 

![](gifs/arch_reg_human.gif "diagonal placement")

## Spot Troubleshooting notes

You can build `spot` from source, official git [repo](https://gitlab.lrde.epita.fr/spot/spot) or Debian package. If you do source installation, then run the following command to verify your installation

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

Please refer to the README file in the tar ball or on their GitHub [page](https://gitlab.lrde.epita.fr/spot/spot/-/blob/next/README) for more details on trouble shooting and installation.


# Citing

If the code is useful in your research, and you would like to acknowledge it, please cite one of the following paper 

- Let's Collaborate: Regret-based Reactive Synthesis for Robotic Manipulation (Explicit Approach) ([paper](https://ieeexplore.ieee.org/document/9812298)):

```
@INPROCEEDINGS{muvvala2022regret,
  author={Muvvala, Karan and Amorese, Peter and Lahijanian, Morteza},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Let's Collaborate: Regret-based Reactive Synthesis for Robotic Manipulation}, 
  year={2022},
  pages={4340-4346},
  doi={10.1109/ICRA46639.2022.9812298}}
```

- Efficient Symbolic Approaches for Quantitative Reactive Synthesis with Finite Tasks (Symbolic Approach) ([paper](https://arxiv.org/abs/2303.03686))

```
@article{muvvala2023efficient,
  title={Efficient Symbolic Approaches for Quantitative Reactive Synthesis with Finite Tasks},
  author={Muvvala, Karan and Lahijanian, Morteza},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  year={2023},
  pages={8666-8672},
  doi={10.1109/IROS55552.2023.10342496}}
}
```

# Contact

Please contact me if you have questions at: karan.muvvala@colorado.edu