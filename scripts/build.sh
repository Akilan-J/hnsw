#!/usr/bin/env bash
# Compiles everything into build/. No external dependencies, so plain javac is
# enough - there is no Maven or Gradle here on purpose.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build
mkdir -p build
javac -Xlint:all -Werror -d build $(find src -name '*.java')
echo "built $(find build -name '*.class' | wc -l | tr -d ' ') classes into build/"
