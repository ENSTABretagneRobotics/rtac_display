# rtac_display

Light-weight display utilities for the RTAC project (extracted from rtac_optix repository because of weak dependency).

## Installation

This package is based on the
[rtac_base](https://gitlab.ensta-bretagne.fr/narvorpi/rtac_base) package
which holds basic types (Matrices, Poses...).

### Dependencies

```
sudo apt install libglfw3-dev libglew-dev
```
A CUDA installation is prefered but not mandatory.

Install the [rtac_base](https://gitlab.ensta-bretagne.fr/narvorpi/rtac_base)
package.



### Installation

This is a regular modern CMake project (uses and exports CMake targets).

Use the -DWITH_CUDA=OFF option if you do not want CUDA related features (which
are of interest only if your project uses CUDA anyway).

```
mkdir build && cd build
cmake -DWITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<an install location in $CMAKE_PREFIX_PATH, or on the system> ..
make -j4 install
```

