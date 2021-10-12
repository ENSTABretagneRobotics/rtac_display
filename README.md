# rtac_display

Light-weight display utilities for the RTAC project (extracted from rtac_optix repository because of weak dependency).

## Installation

This package is based on the
[rtac_base](https://gitlab.ensta-bretagne.fr/narvorpi/rtac_base) package
which holds basic types.

### Dependencies

```
sudo apt install libglfw3-dev libglew-dev
```
A CUDA installation is prefered but not mandatory.

Install the [rtac_base](https://gitlab.ensta-bretagne.fr/narvorpi/rtac_base)
package.



### Installation

This is a regular modern CMake project (uses and exports CMake targets).

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<an install location in $CMAKE_PREFIX_PATH, or on the system> ..
make -j4 install
```

