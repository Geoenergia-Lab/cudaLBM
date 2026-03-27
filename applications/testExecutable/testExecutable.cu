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

    objectRegistry<VelocitySet> runTimeObjects(hostWriteBuffer, mesh, rho, u, v, w, mxx, mxy, mxz, myy, myz, mzz, streamsLBM, programCtrl);

    BlockHalo blockHalo(mesh, programCtrl);

    programCtrl.configure<smem_alloc_size<VelocitySet>()>(kernel::momentBasedLBM);

    const runTimeIO IO(mesh, programCtrl);

    for (host::label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
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
            for (host::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
            {
                hostWriteBuffer.copyFromDevice(
                    device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t>{
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

            postProcess::LBMBin::write(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                functionObjects::solutionVariableNames,
                hostWriteBuffer.data(),
                timeStep);

            runTimeObjects.save(timeStep);
        }

        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        // Main kernel
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            streamsLBM.synchronize(VirtualDeviceIndex);

            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs{
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

            // Configure the kernel to run per GPU
            kernel::momentBasedLBM<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size<VelocitySet>(), streamsLBM.streams()[VirtualDeviceIndex]>>>(devPtrs, readBuffer, writeBuffer);

            errorHandler::checkLast();
        }

        // Sync all devices and streams
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        // Set the device
        errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[0]));

        const host::label_t nxb = mesh.nBlocks<axis::X>();
        const host::label_t nyb = mesh.nBlocks<axis::Y>();

        constexpr const host::threadLabel threadStart(static_cast<device::label_t>(0), static_cast<device::label_t>(0), static_cast<device::label_t>(0));

        const host::label_t Size = static_cast<host::label_t>(sizeof(scalar_t)) * VelocitySet::QF<host::label_t>() * block::nx<host::label_t>() * block::ny<host::label_t>() * mesh.blocksPerDevice<axis::X>() * mesh.blocksPerDevice<axis::Y>();

        constexpr const host::label_t WestDevice = 0;
        constexpr const host::label_t EastDevice = 1;

        constexpr const host::label_t WestPtr_x0 = 4;
        constexpr const host::label_t EastPtr_x1 = 5;

        // East to West exchange
        // Destination z block: located at bz = nzBlocks
        // Pretty sure this is right, not 100%
        const host::blockLabel WestDeviceDestinationBlock(0, 0, 0);
        const host::label_t WestDestinationID = host::idxPop<axis::Z, VelocitySet::QF()>(0, threadStart, WestDeviceDestinationBlock, nxb, nyb);

        // Source z block: located at bz = 0
        // Pretty sure this is right
        const host::blockLabel EastDeviceSourceBlock(0, 0, 0);
        const host::label_t EastSourceID = host::idxPop<axis::Z, VelocitySet::QF()>(0, threadStart, EastDeviceSourceBlock, nxb, nyb);

        errorHandler::check(cudaMemcpyPeer(
            &(blockHalo.writeBuffer(WestDevice).ptr<WestPtr_x0>()[WestDestinationID]),
            programCtrl.deviceList()[WestDevice],
            &(blockHalo.writeBuffer(EastDevice).ptr<WestPtr_x0>()[EastSourceID]),
            programCtrl.deviceList()[EastDevice],
            Size));

        // West to East exchange
        // Destination z block: located at bz = 0
        // Pretty sure this is right
        const host::blockLabel EastDeviceDestinationBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
        const host::label_t EastDestinationID = host::idxPop<axis::Z, VelocitySet::QF()>(0, threadStart, EastDeviceDestinationBlock, nxb, nyb);

        // Source z block: located at bz = nzBlocks
        // Pretty sure this is right
        const host::blockLabel WestDeviceSourceBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
        const host::label_t WestSourceID = host::idxPop<axis::Z, VelocitySet::QF()>(0, threadStart, WestDeviceSourceBlock, nxb, nyb);

        errorHandler::check(cudaMemcpyPeer(
            &(blockHalo.writeBuffer(EastDevice).ptr<EastPtr_x1>()[EastDestinationID]),
            programCtrl.deviceList()[EastDevice],
            &(blockHalo.writeBuffer(WestDevice).ptr<EastPtr_x1>()[WestSourceID]),
            programCtrl.deviceList()[WestDevice],
            Size));

        // Sync all devices and streams
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        // Halo pointer swap
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaDeviceSynchronize());
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            blockHalo.swap(VirtualDeviceIndex);
            errorHandler::checkInline(cudaDeviceSynchronize());
        }
    }

    return 0;
}
