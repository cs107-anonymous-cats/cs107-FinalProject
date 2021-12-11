#!/usr/bin/env bash

tests=(
    test_AutoDiff.py
)


unit='-m pytest'
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="${@} ${unit}"
elif [[ $# -gt 0 && ${1} == 'pytest'* ]]; then
    driver="${@}"
else
    driver="python ${@} ${unit}"
fi

export PYTHONPATH="$(pwd - P)/../src/autodiff":${PYTHONPATH}

${driver} ${tests[@]}
