#!/usr/bin/env bash

poetry run pytest -s -rs -vv \
                  --cov=satdl\
                  --cov-report=term-missing\
                  --cov-report=xml\
                  --log-cli-level=INFO\
                  satdl "${@}"

