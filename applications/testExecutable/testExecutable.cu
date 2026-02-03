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

    // Set cuda device
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaSetDevice(programCtrl.deviceList()[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    const host::latticeMesh mesh(programCtrl);

    VelocitySet::print();

    // Number of mesh points per GPU
    const label_t nxGPUs = mesh.nDevices<axis::X>();
    const label_t nyGPUs = mesh.nDevices<axis::Y>();
    const label_t nzGPUs = mesh.nDevices<axis::Z>();
    // const label_t nGPUs = nxGPUs * nyGPUs * nzGPUs;

    const label_t nxPointsPerGPU = mesh.nx() / nxGPUs;
    const label_t nyPointsPerGPU = mesh.ny() / nyGPUs;
    const label_t nzPointsPerGPU = mesh.nz() / nzGPUs;
    const label_t nPointsPerGPU = nxPointsPerGPU * nyPointsPerGPU * nzPointsPerGPU;

    // Number of mesh blocks per GPU
    const label_t nxBlocksPerGPU = (mesh.nxBlocks()) / nxGPUs; // > Set to device::NUM_BLOCK_X
    const label_t nyBlocksPerGPU = (mesh.nyBlocks()) / nyGPUs; // > Set to device::NUM_BLOCK_Y
    const label_t nzBlocksPerGPU = (mesh.nzBlocks()) / nzGPUs; // > Set to device::NUM_BLOCK_Z
    const dim3 gridBlock{static_cast<uint32_t>(nxBlocksPerGPU), static_cast<uint32_t>(nyBlocksPerGPU), static_cast<uint32_t>(nzBlocksPerGPU)};

    // Create a host array corresponding to the deviceID - now in GPU-major order
    host::array<host::PINNED, label_t, VelocitySet, time::instantaneous> deviceIndexArray(mesh.nPoints(), mesh);

    // Vector of pointers to device memory
    host::array<host::PINNED, label_t *, VelocitySet, time::instantaneous> devicePtrs(nxGPUs * nyGPUs * nzGPUs, nullptr, mesh);

    device::array<field::FULL_FIELD, label_t, VelocitySet, time::instantaneous> testArray(deviceIndexArray);

    // Initialize deviceIndexArray in GPU-major order (all points for GPU 0, then GPU 1, etc.)
    // This makes each GPU's data contiguous in memory
    // for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    // {
    //     for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
    //     {
    //         for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
    //         {
    //             const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;
    //             const label_t startIndex = virtualDeviceIndex * nPointsPerGPU;

    //             // Fill this GPU's contiguous segment
    //             grid_for(
    //                 nxBlocksPerGPU, nyBlocksPerGPU, nzBlocksPerGPU,
    //                 [&](const label_t bx, const label_t by, const label_t bz,
    //                     const label_t tx, const label_t ty, const label_t tz)
    //                 {
    //                     // Local index within GPU (same as kernel expects)
    //                     const label_t localIdx = host::idx(tx, ty, tz, bx, by, bz, nxBlocksPerGPU, nyBlocksPerGPU);

    //                     // Store in GPU-major order: GPU offset + local index
    //                     deviceIndexArray[startIndex + localIdx] = virtualDeviceIndex;
    //                 });
    //         }
    //     }
    // }

    // // Now copy each GPU's contiguous segment to device memory
    // for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    // {
    //     for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
    //     {
    //         for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
    //         {
    //             const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;
    //             const label_t startIndex = virtualDeviceIndex * nPointsPerGPU;

    //             // Allocate memory on the GPU
    //             checkCudaErrors(cudaSetDevice(static_cast<int>(programCtrl.deviceList()[std::min(virtualDeviceIndex, static_cast<label_t>(programCtrl.deviceList().size() - 1))])));
    //             checkCudaErrors(cudaMalloc(&(devicePtrs[virtualDeviceIndex]), nPointsPerGPU * sizeof(label_t)));
    //             std::cout << "GPU " << virtualDeviceIndex << ": Allocated " << nPointsPerGPU << " elements of size " << sizeof(label_t) << std::endl;

    //             // Copy the contiguous segment directly to GPU
    //             // No packing needed - it's already contiguous!
    //             checkCudaErrors(cudaMemcpy(devicePtrs[virtualDeviceIndex], &(deviceIndexArray[startIndex]), nPointsPerGPU * sizeof(label_t), cudaMemcpyHostToDevice));

    //             // Create stream and launch test kernel
    //             const streamHandler<1> streamsLBM;
    //             testKernel<<<gridBlock, mesh.threadBlock(), 0, streamsLBM.streams()[0]>>>(devicePtrs[virtualDeviceIndex], nxBlocksPerGPU, nyBlocksPerGPU, (GPU_x * nxBlocksPerGPU), (GPU_y * nyBlocksPerGPU), (GPU_z * nzBlocksPerGPU), virtualDeviceIndex);
    //             checkCudaErrors(cudaDeviceSynchronize());
    //         }
    //     }
    // }

    for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    {
        for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
        {
            for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
            {
                const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;

                // Create stream and launch test kernel
                const streamHandler<1> streamsLBM;
                testKernel<<<gridBlock, mesh.threadBlock(), 0, streamsLBM.streams()[0]>>>(devicePtrs[virtualDeviceIndex], nxBlocksPerGPU, nyBlocksPerGPU, (GPU_x * nxBlocksPerGPU), (GPU_y * nyBlocksPerGPU), (GPU_z * nzBlocksPerGPU), virtualDeviceIndex);
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }
    }

    // Copy back from GPU memory to the same contiguous segments
    for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    {
        for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
        {
            for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
            {
                const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;
                const label_t startIndex = virtualDeviceIndex * nPointsPerGPU;

                // Set the active device
                checkCudaErrors(cudaSetDevice(static_cast<int>(programCtrl.deviceList()[std::min(virtualDeviceIndex, static_cast<label_t>(programCtrl.deviceList().size() - 1))])));
                checkCudaErrors(cudaDeviceSynchronize());

                // Copy back from device to the contiguous segment
                checkCudaErrors(cudaMemcpy(&(deviceIndexArray[startIndex]), devicePtrs[virtualDeviceIndex], nPointsPerGPU * sizeof(label_t), cudaMemcpyDeviceToHost));

                checkCudaErrors(cudaDeviceSynchronize());
            }
        }
    }

    // After reconstruction, verify the data
    bool verificationFailed = false;

    // Verify each GPU's segment
    for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    {
        for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
        {
            for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
            {
                const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;
                const label_t expectedValue = virtualDeviceIndex + 100;
                const label_t startIndex = virtualDeviceIndex * nPointsPerGPU;

                // Check all points in this GPU's segment
                for (label_t i = 0; i < nPointsPerGPU; i++)
                {
                    if (deviceIndexArray[startIndex + i] != expectedValue)
                    {
                        // Convert local index to coordinates for debugging
                        label_t localIdx = i;
                        const label_t tx = localIdx % block::nx();
                        localIdx /= block::nx();
                        const label_t ty = localIdx % block::ny();
                        localIdx /= block::ny();
                        const label_t tz = localIdx % block::nz();
                        localIdx /= block::nz();
                        const label_t bx = localIdx % nxBlocksPerGPU;
                        localIdx /= nxBlocksPerGPU;
                        const label_t by = localIdx % nyBlocksPerGPU;
                        const label_t bz = localIdx / nyBlocksPerGPU;

                        const label_t global_bx = bx + (GPU_x * nxBlocksPerGPU);
                        const label_t global_by = by + (GPU_y * nyBlocksPerGPU);
                        const label_t global_bz = bz + (GPU_z * nzBlocksPerGPU);

                        std::cout << "Verification failed for GPU " << virtualDeviceIndex
                                  << " at (block: " << global_bx << "," << global_by << "," << global_bz
                                  << " thread: " << tx << "," << ty << "," << tz << ")"
                                  << ": expected " << expectedValue
                                  << ", got " << deviceIndexArray[startIndex + i] << std::endl;
                        verificationFailed = true;
                        {
                            break;
                        }
                    }
                }
                if (verificationFailed)
                {
                    break;
                }
            }
            if (verificationFailed)
            {
                break;
            }
        }
        if (verificationFailed)
        {
            break;
        }
    }

    if (!verificationFailed)
    {
        std::cout << "Reconstruction verification passed!" << std::endl;
        std::cout << std::endl;
    }

    // Create a temporary 2D array to reconstruct the z=0 plane
    std::vector<std::vector<label_t>> plane(mesh.ny(), std::vector<label_t>(mesh.nx(), 999));

    for (label_t GPU_z = 0; GPU_z < nzGPUs; GPU_z++)
    {
        for (label_t GPU_y = 0; GPU_y < nyGPUs; GPU_y++)
        {
            for (label_t GPU_x = 0; GPU_x < nxGPUs; GPU_x++)
            {
                const label_t virtualDeviceIndex = GPU_x + GPU_y * nxGPUs + GPU_z * nxGPUs * nyGPUs;
                const label_t startIndex = virtualDeviceIndex * nPointsPerGPU;

                grid_for(
                    nxBlocksPerGPU, nyBlocksPerGPU, nzBlocksPerGPU,
                    [&](const label_t bx, const label_t by, const label_t bz,
                        const label_t tx, const label_t ty, const label_t tz)
                    {
                        // Calculate global coordinates
                        const label_t x = tx + block::nx() * (bx + (GPU_x * nxBlocksPerGPU));
                        const label_t y = ty + block::ny() * (by + (GPU_y * nyBlocksPerGPU));
                        const label_t z = tz + block::nz() * (bz + (GPU_z * nzBlocksPerGPU));

                        if (z == 0)
                        {
                            // Calculate local index
                            const label_t localIdx = host::idx(tx, ty, tz, bx, by, bz, nxBlocksPerGPU, nyBlocksPerGPU);
                            plane[y][x] = deviceIndexArray[startIndex + localIdx] - 100; // Subtract 100 to get original GPU ID
                        }
                    });
            }
        }
    }

    // Print the plane
    for (label_t y = 0; y < mesh.ny(); y++)
    {
        for (label_t x = 0; x < mesh.nx(); x++)
        {
            std::cout << plane[y][x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

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