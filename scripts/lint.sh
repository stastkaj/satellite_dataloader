#!/usr/bin/env bash

poetry run mypy --install-types --non-interactive satdl
poetry run flake8 satdl

