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
    const label_t nPointsPerGPU = nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU;

    // Number of mesh blocks per GPU
    const label_t nxBlocksPerGPU = (mesh.nxBlocks()) / nxGPUs; // > Set to device::NUM_BLOCK_X
    const label_t nyBlocksPerGPU = (mesh.nyBlocks()) / nyGPUs; // > Set to device::NUM_BLOCK_Y
    const label_t nzBlocksPerGPU = (mesh.nzBlocks()) / nzGPUs; // > Set to device::NUM_BLOCK_Z
    const dim3 gridBlock{static_cast<uint32_t>(nxBlocksPerGPU), static_cast<uint32_t>(nyBlocksPerGPU), static_cast<uint32_t>(nzBlocksPerGPU)};

    // Create a host array corresponding to the deviceID
    host::array<host::PINNED, label_t, VelocitySet, time::instantaneous> deviceIndexArray(mesh.nPoints());

    // Vector of pointers to device memory
    host::array<host::PINNED, label_t *, VelocitySet, time::instantaneous> devicePtrs(nxGPUs * nyGPUs * nzGPUs, nullptr);

    // Temporary vector used for partitioning the domain
    std::vector<label_t> temp(nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU, 0);

    for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    {
        for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
        {
            for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
            {
                // Get the device index
                const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;

                // Define the test array for this partition of the GPU
                grid_for(
                    nxBlocksPerGPU, nyBlocksPerGPU, nzBlocksPerGPU,
                    [&](const label_t bx, const label_t by, const label_t bz,
                        const label_t tx, const label_t ty, const label_t tz)
                    {
                        // Global coordinates
                        const label_t x = tx + (block::nx() * (bx + (GPU_x * nxBlocksPerGPU)));
                        const label_t y = ty + (block::ny() * (by + (GPU_y * nyBlocksPerGPU)));
                        const label_t z = tz + (block::nz() * (bz + (GPU_z * nzBlocksPerGPU)));

                        // Linear index for the domain point
                        const label_t linearIndex = host::idx(tx, ty, tz, bx + (GPU_x * nxBlocksPerGPU), by + (GPU_y * nyBlocksPerGPU), bz + (GPU_z * nzBlocksPerGPU), mesh);
                        deviceIndexArray[linearIndex] = virtualDeviceIndex;
                    });

                // Load this partition of the domain into the temporary contiguous buffer
                grid_for(
                    nxBlocksPerGPU, nyBlocksPerGPU, nzBlocksPerGPU,
                    [&](const label_t bx, const label_t by, const label_t bz,
                        const label_t tx, const label_t ty, const label_t tz)
                    {
                        // Calculate the index in the temp buffer
                        const label_t I = host::idx(tx, ty, tz, bx, by, bz, nxBlocksPerGPU, nyBlocksPerGPU);

                        // Calculate the index in the global buffer
                        const label_t i = host::idx(tx, ty, tz, bx + (GPU_x * nxBlocksPerGPU), by + (GPU_y * nyBlocksPerGPU), bz + (GPU_z * nzBlocksPerGPU), mesh);

                        // Copy to temp buffer from global
                        temp[I] = deviceIndexArray[i];
                    });

                // Allocate memory on the GPU
                checkCudaErrors(cudaSetDevice(static_cast<int>(programCtrl.deviceList()[virtualDeviceIndex])));
                checkCudaErrors(cudaMalloc(&(devicePtrs[virtualDeviceIndex]), nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU * sizeof(label_t)));
                std::cout << "Allocated " << nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU << " elements of size " << sizeof(label_t) << std::endl;

                // Copy the temporary buffer to the GPU
                device::copy((devicePtrs[virtualDeviceIndex]), temp);

                // Create stream and launch test kernel
                const streamHandler<1> streamsLBM;
                testKernel<<<gridBlock, mesh.threadBlock(), 0, streamsLBM.streams()[0]>>>((devicePtrs[virtualDeviceIndex]), nxBlocksPerGPU, nyBlocksPerGPU, (GPU_x * nxBlocksPerGPU), (GPU_y * nyBlocksPerGPU), (GPU_z * nzBlocksPerGPU), virtualDeviceIndex);
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }
    }

    // Attempt to reconstruct from GPU memory
    for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    {
        for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
        {
            for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
            {
                // Get the device index
                const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;

                // Set the active device
                checkCudaErrors(cudaSetDevice(static_cast<int>(programCtrl.deviceList()[virtualDeviceIndex])));

                checkCudaErrors(cudaDeviceSynchronize());

                // Copy to the temporary buffer
                host::to_host(devicePtrs[virtualDeviceIndex], temp.data(), 0, nPointsPerGPU);

                checkCudaErrors(cudaDeviceSynchronize());

                // Place back into host buffer
                grid_for(
                    nxBlocksPerGPU, nyBlocksPerGPU, nzBlocksPerGPU,
                    [&](const label_t bx, const label_t by, const label_t bz,
                        const label_t tx, const label_t ty, const label_t tz)
                    {
                        // Calculate the index in the temp buffer
                        const label_t I = host::idx(tx, ty, tz, bx, by, bz, nxBlocksPerGPU, nyBlocksPerGPU);

                        // Calculate the index in the global buffer
                        const label_t i = host::idx(tx, ty, tz, bx + (GPU_x * nxBlocksPerGPU), by + (GPU_y * nyBlocksPerGPU), bz + (GPU_z * nzBlocksPerGPU), mesh);

                        // Copy from temp to global
                        deviceIndexArray[i] = temp[I];
                    });
            }
        }
    }

    // After reconstruction, verify the data
    bool verificationFailed = false;

    grid_for(
        mesh.nxBlocks(), mesh.nyBlocks(), mesh.nzBlocks(),
        [&](const label_t bx, const label_t by, const label_t bz,
            const label_t tx, const label_t ty, const label_t tz)
        {
            const label_t idx = host::idx(tx, ty, tz, bx, by, bz, mesh);

            // Calculate which GPU this point belongs to
            const label_t gpu_x = bx / nxBlocksPerGPU;
            const label_t gpu_y = by / nyBlocksPerGPU;
            const label_t gpu_z = bz / nzBlocksPerGPU;

            // Only check if within valid GPU ranges
            if (gpu_x < nxGPUs && gpu_y < nyGPUs && gpu_z < nzGPUs)
            {
                const label_t expectedGPU = gpu_x + gpu_y * nxGPUs + gpu_z * nxGPUs * nyGPUs;
                const label_t expectedValue = expectedGPU + 100;

                if (deviceIndexArray[idx] != expectedValue)
                {
                    std::cout << "Verification failed at ("
                              << tx << "," << ty << "," << tz
                              << ") in block ("
                              << bx << "," << by << "," << bz
                              << "): expected " << expectedValue
                              << ", got " << deviceIndexArray[idx]
                              << std::endl;
                    verificationFailed = true;
                }
            }
        });

    if (!verificationFailed)
    {
        std::cout << "Reconstruction verification passed!" << std::endl;
    }

    // Clean up memory used for testing
    for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    {
        for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
        {
            for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
            {
                const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;
                std::cout << "Freeing memory from address " << devicePtrs[virtualDeviceIndex] << " on device " << virtualDeviceIndex << std::endl;
                checkCudaErrors(cudaFree(devicePtrs[virtualDeviceIndex]));
            }
        }
    }

    return 0;
}