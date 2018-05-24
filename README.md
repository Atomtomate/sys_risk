Dependencies:
- Eigen3
- trng (Tina's random number generator)
- MPI
- Boost
- (currently optional) SUNDIALS
- Stan Math (included)
- (currently optional) RInside/RCpp

optional:
- GTest

Installation on ubuntu:
    - git clone https://github.com/Atomtomate/sys_risk && cd sys_risk
    - sudo apt-get update -qq
    - sudo apt-get install build-essential autotools-dev mercurial libeigen3-dev libboost-all-dev libgtest-dev google-mock git libblas-dev liblapack-dev
    - mkdir build && cd build && cmake ..
    - configure cmake as needed
    - make



Install with: mkdir build && cd build && cmake .. && make
