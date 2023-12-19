g++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` ./checkmate_lib.cpp -o checkmate_lib`python3-config --extension-suffix`
g++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` ./checkmate_lib_old.cpp -o checkmate_lib_old`python3-config --extension-suffix`
g++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` ./checkmate_objective_lib.cpp -o checkmate_objective_lib`python3-config --extension-suffix`
g++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` ./naotti/AI.cpp -o naotti2020`python3-config --extension-suffix`