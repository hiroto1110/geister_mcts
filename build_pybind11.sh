g++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` ./geister_lib.cpp -o geister_lib`python3-config --extension-suffix`