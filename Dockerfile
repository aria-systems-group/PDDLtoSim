FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get -y update && apt -y install \
    gcc git \
    wget vim curl \
    python3-pip cmake automake \
    build-essential \
    apt-utils flex bison mona \
    libjs-mathjax libjs-requirejs

# installing spot using apt
RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add -

SHELL ["/bin/bash", "-c"] 

RUN pip3 install pyyaml numpy<2.0 bidict networkx graphviz ply pybullet \
    pyperplan==1.3 IPython svgwrite matplotlib==3.5 imageio lark-parser==0.9.0 sympy==1.6.1 \
    cloudpickle cycler future mpmath pandas pydot pyglet pytz scipy \
    gym==0.21.0 gym_minigrid==1.0.2 joblib tqdm shapely paramiko

# TODO do I need pydot3?

# doing this on a separate line as pydot3 uses a command in setup.py compatable with setuptools<=58
RUN pip3 install setuptools==65.5.0


# LATEST Spot 2.12 seems to have come issues with libltdl version
# RUN echo 'deb [trusted=true] http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list

# Based on https://gitlab.lre.epita.fr/spot/spot/-/issues/544#note_29198
RUN echo 'deb [trusted=true] https://download.opensuse.org/repositories/home:/adl/xUbuntu_22.04/ ./' >/etc/apt/sources.list


RUN apt-get -y update && apt -y install spot \
    libspot-dev \
    spot-doc

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ADD ./ /root/pddltosim

RUN cd /root/pddltosim/regret_synthesis_toolbox/src/LTLf2DFA && pip3 install .

WORKDIR /root/pddltosim

ENTRYPOINT "/bin/bash"