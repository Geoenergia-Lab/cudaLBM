/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Post-processing utility to calculate derived fields from saved moment fields
    Supported calculations: velocity magnitude, velocity divergence, vorticity,
    vorticity magnitude, integrated vorticity

Namespace
    LBM

SourceFiles
    testExecutable.cu

\*---------------------------------------------------------------------------*/

#include "testExecutable.cuh"

using namespace LBM;

using VelocitySet = D3Q19;

constexpr const label_t nxGPUs = 1;
constexpr const label_t nyGPUs = 1;
constexpr const label_t nzGPUs = 2;

int main(const int argc, const char *const argv[])
{
    static_assert((std::is_same<BoundaryConditions, lidDrivenCavity>::value) || std::is_same<BoundaryConditions, jetFlow>::value);

    const programControl programCtrl(argc, argv);

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    const host::latticeMesh mesh(programCtrl);

    VelocitySet::print();

    // Number of mesh points per GPU
    const label_t nx = mesh.nx() / nxGPUs;
    const label_t ny = mesh.ny() / nyGPUs;
    const label_t nz = mesh.nz() / nzGPUs;

    // Loop over the GPUs and print the offsets
    for (label_t z = 0; z < nzGPUs; z++)
    {
        for (label_t y = 0; y < nyGPUs; y++)
        {
            for (label_t x = 0; x < nxGPUs; x++)
            {
                const label_t deviceID = x + y * nxGPUs + z * nxGPUs * nyGPUs;
                const label_t bx = (x * nx) / block::nx();
                const label_t by = (y * ny) / block::ny();
                const label_t bz = (z * nz) / block::nz();
                const device::latticeMesh devMesh(deviceID, nx, ny, nz, bx, by, bz);
                devMesh.print();
                std::cout << std::endl;
            }
        }
    }

    // label_t *ptr_0;

    // checkCudaErrors(cudaMalloc(&ptr_0, nx * ny * nz * sizeof(label_t)));
    // std::cout << "Allocated " << nx * ny * nz << " elements" << std::endl;
    // checkCudaErrors(cudaFree(ptr_0));

    return 0;
}