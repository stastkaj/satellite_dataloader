#!/usr/bin/env bash

poetry run autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place satdl --exclude=__init__.py
poetry run isort satdl
poetry run black satdl

