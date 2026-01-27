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

int main(const int argc, const char *const argv[])
{
    static_assert((std::is_same<BoundaryConditions, lidDrivenCavity>::value) || std::is_same<BoundaryConditions, jetFlow>::value);

    const programControl programCtrl(argc, argv);

    const label_t nxGPUs = string::extractParameter<label_t>(string::readFile("deviceDecomposition"), "nx");
    const label_t nyGPUs = string::extractParameter<label_t>(string::readFile("deviceDecomposition"), "ny");
    const label_t nzGPUs = string::extractParameter<label_t>(string::readFile("deviceDecomposition"), "nz");

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    const host::latticeMesh mesh(programCtrl);

    VelocitySet::print();

    // Number of mesh points per GPU
    const label_t nxPointsPerGPU = mesh.nx() / nxGPUs;
    const label_t nyPointsPerGPU = mesh.ny() / nyGPUs;
    const label_t nzPointsPerGPU = mesh.nz() / nzGPUs;

    // Number of mesh blocks per GPU
    const label_t nxBlocksPerGPU = (mesh.nxBlocks()) / nxGPUs; // > Set to device::NUM_BLOCK_X
    const label_t nyBlocksPerGPU = (mesh.nyBlocks()) / nyGPUs; // > Set to device::NUM_BLOCK_Y
    const label_t nzBlocksPerGPU = (mesh.nzBlocks()) / nzGPUs; // > Set to device::NUM_BLOCK_Z
    const dim3 gridBlock{static_cast<uint32_t>(nxBlocksPerGPU), static_cast<uint32_t>(nyBlocksPerGPU), static_cast<uint32_t>(nzBlocksPerGPU)};

    // Create a host array corresponding to the deviceID
    host::array<host::PINNED, label_t, VelocitySet, time::instantaneous> deviceIndexArray(mesh.nPoints());

    for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    {
        for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
        {
            for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
            {
                const label_t correctDevice = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;
                for (label_t bz = 0; bz < nzBlocksPerGPU; bz++)
                {
                    for (label_t by = 0; by < nyBlocksPerGPU; by++)
                    {
                        for (label_t bx = 0; bx < nxBlocksPerGPU; bx++)
                        {
                            for (label_t tz = 0; tz < block::nz(); tz++)
                            {
                                for (label_t ty = 0; ty < block::ny(); ty++)
                                {
                                    for (label_t tx = 0; tx < block::nx(); tx++)
                                    {
                                        const label_t x = tx + (block::nx() * (bx + (GPU_x * nxBlocksPerGPU)));
                                        const label_t y = ty + (block::ny() * (by + (GPU_y * nyBlocksPerGPU)));
                                        const label_t z = tz + (block::nz() * (bz + (GPU_z * nzBlocksPerGPU)));

                                        // Determine which partition this point belongs to
                                        const label_t partitionX = x / nxPointsPerGPU;
                                        const label_t partitionY = y / nyPointsPerGPU;
                                        const label_t partitionZ = z / nzPointsPerGPU;

                                        // Calculate device ID for this partition
                                        const label_t deviceID = partitionX + partitionY * nxGPUs + partitionZ * nxGPUs * nyGPUs;

                                        // Linear index for the domain point
                                        const label_t linearIndex = host::idx(tx, ty, tz, bx + (GPU_x * nxBlocksPerGPU), by + (GPU_y * nyBlocksPerGPU), bz + (GPU_z * nzBlocksPerGPU), mesh);
                                        deviceIndexArray[linearIndex] = deviceID;
                                    }
                                }
                            }
                        }
                    }
                }

                // Construct a temporary vector
                std::vector<label_t> temp(nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU, 0);
                label_t i = 0;
                for (label_t bz = 0; bz < nzBlocksPerGPU; bz++)
                {
                    for (label_t by = 0; by < nyBlocksPerGPU; by++)
                    {
                        for (label_t bx = 0; bx < nxBlocksPerGPU; bx++)
                        {
                            for (label_t tz = 0; tz < block::nz(); tz++)
                            {
                                for (label_t ty = 0; ty < block::ny(); ty++)
                                {
                                    for (label_t tx = 0; tx < block::nx(); tx++)
                                    {
                                        temp[i] = deviceIndexArray[host::idx(tx, ty, tz, bx + (GPU_x * nxBlocksPerGPU), by + (GPU_y * nyBlocksPerGPU), bz + (GPU_z * nzBlocksPerGPU), mesh)];
                                        i++;
                                    }
                                }
                            }
                        }
                    }
                }

                label_t *ptr_0;

                checkCudaErrors(cudaMalloc(&ptr_0, nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU * sizeof(label_t)));
                std::cout << "Allocated " << nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU << " elements of size " << sizeof(label_t) << std::endl;
                device::copy(ptr_0, temp);

                const streamHandler<1> streamsLBM;

                testKernel<<<gridBlock, mesh.threadBlock(), 0, streamsLBM.streams()[0]>>>(ptr_0, nxBlocksPerGPU, nyBlocksPerGPU, (GPU_x * nxBlocksPerGPU), (GPU_y * nyBlocksPerGPU), (GPU_z * nzBlocksPerGPU), correctDevice);

                checkCudaErrors(cudaFree(ptr_0));
            }
        }
    }

    return 0;
}