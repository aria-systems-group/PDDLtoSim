from ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get -y update && apt -y install \
    gcc git \
    wget vim curl \
    python3-pip cmake automake \
    build-essential \
    apt-utils flex bison mona 

# installing spot using apt
RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add -

SHELL ["/bin/bash", "-c"] 

RUN pip3 install pyyaml numpy bidict networkx graphviz ply pybullet \
    pyperplan==1.3 IPython svgwrite matplotlib imageio lark-parser==0.9.0 sympy==1.6.1 \
    cloudpickle cycler future mpmath pandas pydot pydot3 pyglet pytz scipy \
    gym gym_minigrid joblib tqdm shapely paramiko

RUN echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list

RUN apt-get -y update && apt -y install spot \
    libspot-dev \
    spot-doc

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ADD ./ /root/pddltosim

WORKDIR /root/pddltosim

ENTRYPOINT "/bin/bash"