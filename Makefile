
PYTHON=./env/bin/python
CONDA=conda

all: env

setup: env pip chemhelp

env:
	${CONDA} env create -f requirements.yml -p env

pip: env
	${PYTHON} -m pip install numpy
	${PYTHON} -m pip install -r requirements.txt --no-cache-dir

chemhelp:
	git clone https://github.com/charnley/chemhelp

jupyter-pip:
	${PYTHON} -m pip install nglview ipywidgets
	./env/bin/jupyter-nbextension enable --py --sys-prefix widgetsnbextension
	./env/bin/jupyter-nbextension enable --py --sys-prefix nglview

data:
	mkdir -p data

#

notebook:
	env/bin/jupyter notebook

#

clean:
	rm *.pyc __pycache__

super-clean:
	rm -fr data env __pycache__

