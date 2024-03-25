.PHONY: init
init:
	conda create --name kimchi python=3.11 -y

.PHONY: kimchi
kimchi:
	conda env create -f kimchi.yml

.PHONY: setup
setup:
	python setup.py sdist bdist_wheel

.PHONY: upload
upload:
	twine upload dist/*

.PHONY: export
export:
	conda env export > kimchi.yml


.PHONY: exit
exit:
	conda init
	conda deactivate

.PHONY: remove
remove:
	conda env remove --name kimchi -y