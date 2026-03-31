#/bin/bash
set -e

for t in tests/bash/test_*.sh; do
    echo "========================== Testing $t =================================="
    bash $t;
done
