#!/bin/bash

###############################################################################
# This is a revision based on a script taken from the "modulus" repository (Apache 2.0 License)
# Repository: https://github.com/NVIDIA/modulus
# License: Apache 2.0
#
# Description: This script executes coverage checks using `coverage` and `pytest`.
#              It runs the tests, combines the coverage data, generates a coverage
#              report, and cleans up the generated coverage files.
##############################################################################

# do the coverage checks
coverage run \
--rcfile='coverage.pytest.rc' \
-m pytest \

coverage run \
--rcfile='coverage.docstring.rc' \
-m pytest \
--doctest-modules ../src/flunet//

coverage combine --data-file=.coverage
coverage report --omit=*test*

# if you wish to view the report in HTML format uncomment below
coverage html --omit=*test*

# cleanup
rm .coverage
