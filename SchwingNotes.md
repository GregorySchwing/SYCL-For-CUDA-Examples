To run with the minimum setup required:

Will only run on the CPU.

Prerequisite:

DPC++ compiler

https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp

Download online installer
Install
source /opt/intel/oneapi/setvars.sh

compile with
dpcpp -o a.out src/*.cpp

i.e.
dpcpp -o a.out vector_addition.cpp