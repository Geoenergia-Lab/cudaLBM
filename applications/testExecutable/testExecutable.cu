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

template <const label_t N, typename T>
[[nodiscard]] bool collection_contains_nullptr(const device::ptrCollection<N, T> &devPtrs)
{
    for (label_t i = 0; i < N; i++)
    {
        if (devPtrs[i] == nullptr)
        {
            return true;
        }
    }
    return false;
}

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    // Set cuda device
    errorHandler::check(cudaDeviceSynchronize());
    errorHandler::check(cudaSetDevice(programCtrl.deviceList()[0]));
    errorHandler::check(cudaDeviceSynchronize());

    const host::latticeMesh mesh(programCtrl);

    VelocitySet::print();

    // Allocate the arrays on the device
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> rho("rho", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> u("u", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> v("v", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> w("w", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxx("m_xx", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxy("m_xy", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxz("m_xz", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> myy("m_yy", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> myz("m_yz", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mzz("m_zz", mesh, programCtrl);

    // Setup Streams
    const streamHandler streamsLBM(programCtrl);

    // Allocate a buffer of pinned memory on the host for writing
    host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> hostWriteBuffer(mesh.size() * NUMBER_MOMENTS(), mesh);

    // objectRegistry<VelocitySet, NStreams()> runTimeObjects(hostWriteBuffer, mesh, devPtrs, streamsLBM, programCtrl);

    BlockHalo blockHalo(mesh, programCtrl);

    kernel::configure<smem_alloc_size()>(momentBasedD3Q27, programCtrl);

    const runTimeIO IO(mesh, programCtrl);

    for (label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        // Do the run-time IO
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << std::endl;
        }

        // Checkpoint
        if (programCtrl.save(timeStep))
        {
            // Do this in a loop
            for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
            {
                hostWriteBuffer.copy_from_device(
                    device::ptrCollection<10, scalar_t>{
                        rho.ptr(VirtualDeviceIndex),
                        u.ptr(VirtualDeviceIndex),
                        v.ptr(VirtualDeviceIndex),
                        w.ptr(VirtualDeviceIndex),
                        mxx.ptr(VirtualDeviceIndex),
                        mxy.ptr(VirtualDeviceIndex),
                        mxz.ptr(VirtualDeviceIndex),
                        myy.ptr(VirtualDeviceIndex),
                        myz.ptr(VirtualDeviceIndex),
                        mzz.ptr(VirtualDeviceIndex)},
                    mesh,
                    VirtualDeviceIndex);
            }

            fileIO::writeFile<time::instantaneous>(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                functionObjects::solutionVariableNames,
                hostWriteBuffer.data(),
                timeStep,
                rho.meanCount());

            // runTimeObjects.save(timeStep);
        }

        for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        // Main kernel
        for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            streamsLBM.synchronize(VirtualDeviceIndex);

            // errorHandler::checkInline(cudaStreamSynchronize(streamsLBM.streams()[VirtualDeviceIndex]));

            // if (programCtrl.print(timeStep))
            // {
            //     std::cout << "deviceIdx: " << programCtrl.deviceList()[VirtualDeviceIndex] << std::endl;
            // }

            const device::ptrCollection<10, scalar_t> devPtrs{
                rho.ptr(VirtualDeviceIndex),
                u.ptr(VirtualDeviceIndex),
                v.ptr(VirtualDeviceIndex),
                w.ptr(VirtualDeviceIndex),
                mxx.ptr(VirtualDeviceIndex),
                mxy.ptr(VirtualDeviceIndex),
                mxz.ptr(VirtualDeviceIndex),
                myy.ptr(VirtualDeviceIndex),
                myz.ptr(VirtualDeviceIndex),
                mzz.ptr(VirtualDeviceIndex)};

            const device::ptrCollection<6, const scalar_t> readBuffer = blockHalo.readBuffer(VirtualDeviceIndex);
            const device::ptrCollection<6, scalar_t> writeBuffer = blockHalo.writeBuffer(VirtualDeviceIndex);

            // std::cout << "devPtrs on device " << VirtualDeviceIndex << (collection_contains_nullptr(devPtrs) ? " contains nullptr" : " does not contain nullptr") << std::endl;
            // std::cout << "readBuffer on device " << VirtualDeviceIndex << (collection_contains_nullptr(readBuffer) ? " contains nullptr" : " does not contain nullptr") << std::endl;
            // std::cout << "writeBuffer on device " << VirtualDeviceIndex << (collection_contains_nullptr(writeBuffer) ? " contains nullptr" : " does not contain nullptr") << std::endl;

            // Configure the kernel to run per GPU
            momentBasedD3Q19<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size(), streamsLBM.streams()[VirtualDeviceIndex]>>>(devPtrs, readBuffer, writeBuffer);

            // testKernel<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size(), streamsLBM.streams()[VirtualDeviceIndex]>>>(devPtrs, readBuffer, writeBuffer);

            // testKernel<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[VirtualDeviceIndex]>>>();

            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cout << "Launch error: " << cudaGetErrorString(err) << std::endl;
            }
        }

        // Sync all devices and streams
        for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        if constexpr (true)
        {
            // Set the device
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[0]));

            // Get the pointers
            // GPU 0 needs to pull from the z0 pointer
            // GPU 1 needs to pull from the z1 pointer
            // So it should be GPU 0 > 1, z1
            // GPU 1 > 0, z0
            // When copying from the West boundary of the East GPU, we need to add an offset of bz by +1
            // Also when copying from the East of the West GPU, we need to add an inset of bz by -1
            // constexpr const std::size_t offset = 0; // Has to be idxPop at bz = 1, I think

            // Offset by 0 or 1 into the GPU: These are located on the Front device
            const std::size_t NoOffset = host::idxPop<axis::Z, VelocitySet::QF()>(0, blockLabel_t(0, 0, 0), blockLabel_t(0, 0, 0), mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>());
            const std::size_t Offset = host::idxPop<axis::Z, VelocitySet::QF()>(0, blockLabel_t(0, 0, 0), blockLabel_t(0, 0, 1), mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>());

            // Inset by 0 or 1 into the GPU: These are located on the Back device
            const std::size_t NoInset = host::idxPop<axis::Z, VelocitySet::QF()>(0, blockLabel_t(0, 0, 0), blockLabel_t(0, 0, mesh.blocksPerDevice<axis::Z>() - 1), mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>());
            const std::size_t Inset = host::idxPop<axis::Z, VelocitySet::QF()>(0, blockLabel_t(0, 0, 0), blockLabel_t(0, 0, mesh.blocksPerDevice<axis::Z>() - 2), mesh.nBlocks<axis::X>(), mesh.nBlocks<axis::Y>());

            // OFFSET SHOULD BE ADDING + 1 TO BZ ALWAYS SINCE IT HAS AN EXTRA LAYER!!!

            // constexpr const std::size_t copySize = 128; // Has to be the size of nbx * nby * QF * ntx * nty
            const std::size_t Size = static_cast<std::size_t>(sizeof(scalar_t)) * VelocitySet::QF<std::size_t>() * block::nx<std::size_t>() * block::ny<std::size_t>() * mesh.blocksPerDevice<axis::X, std::size_t>() * mesh.blocksPerDevice<axis::Y, std::size_t>();

            constexpr const label_t BackDevice = 0;  // Should receive z1
            constexpr const label_t FrontDevice = 1; // Should receive z0

            constexpr const label_t BackPtr_z0 = 4;
            constexpr const label_t FrontPtr_z1 = 5;

            // 1. From Back device front face (ptr 5) to Front device back face (ptr 4)
            cudaMemcpyPeer(
                &(blockHalo.writeBuffer(FrontDevice).ptr<FrontPtr_z1>()[NoOffset]),
                programCtrl.deviceList()[FrontDevice],

                &(blockHalo.writeBuffer(BackDevice).ptr<FrontPtr_z1>()[Inset]),
                programCtrl.deviceList()[BackDevice],
                Size);

            // 2. From Front device back face (ptr 4) to Back device front face (ptr 5)
            cudaMemcpyPeer(
                &(blockHalo.writeBuffer(BackDevice).ptr<BackPtr_z0>()[NoInset]),
                programCtrl.deviceList()[BackDevice],

                &(blockHalo.writeBuffer(FrontDevice).ptr<BackPtr_z0>()[Offset]),
                programCtrl.deviceList()[FrontDevice],
                Size);
        }

        // Sync all devices and streams
        for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        // Halo pointer swap
        for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            blockHalo.swap(VirtualDeviceIndex);
        }
    }

    return 0;
}
