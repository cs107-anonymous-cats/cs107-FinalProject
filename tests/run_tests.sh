#!/usr/bin/env bash

tests=(
    test_DualNum.py
)


unit='-m pytest'
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="${@} ${unit}"
elif [[ $# -gt 0 && ${1} == 'pytest'* ]]; then
    driver="${@}"
else
    driver="python ${@} ${unit}"
fi

export PYTHONPATH="$(pwd - P)/../src":${PYTHONPATH}

${driver} ${tests[@]}
