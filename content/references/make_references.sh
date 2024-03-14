#!/bin/bash

pandoc --wrap=none --citeproc --standalone --bibliography=references.bib --csl=ieee.csl --from=markdown --to=commonmark make_references.md_ -o references.md

python fix_references.py
