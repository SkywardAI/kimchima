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