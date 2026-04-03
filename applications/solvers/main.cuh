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
    Common main function definition for all LBM solvers

Namespace
    LBM

SourceFiles
    main.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MAIN_CUH
#define __MBLBM_MAIN_CUH

using namespace LBM;

__host__ [[nodiscard]] inline consteval device::label_t NStreams() noexcept { return 1; }

__host__ [[nodiscard]] inline consteval device::label_t VirtualDeviceIndex() noexcept { return 0; }

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
    device::scalarField<VelocitySet, time::instantaneous> rho("rho", mesh, programCtrl);
    device::vectorField<VelocitySet, time::instantaneous> U("U", mesh, programCtrl);
    device::symmetricTensorField<VelocitySet, time::instantaneous> Pi("Pi", mesh, programCtrl);

    const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs(
        rho.self().ptr(VirtualDeviceIndex()),
        U.x().ptr(VirtualDeviceIndex()),
        U.y().ptr(VirtualDeviceIndex()),
        U.z().ptr(VirtualDeviceIndex()),
        Pi.xx().ptr(VirtualDeviceIndex()),
        Pi.xy().ptr(VirtualDeviceIndex()),
        Pi.xz().ptr(VirtualDeviceIndex()),
        Pi.yy().ptr(VirtualDeviceIndex()),
        Pi.yz().ptr(VirtualDeviceIndex()),
        Pi.zz().ptr(VirtualDeviceIndex()));

    // Setup Streams
    const streamHandler streamsLBM(programCtrl);

    // Allocate a buffer of pinned memory on the host for writing
    host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> hostWriteBuffer(mesh.size() * NUMBER_MOMENTS(), mesh);

    // objectRegistry<VelocitySet> runTimeObjects(hostWriteBuffer, mesh, rho, U, Pi, streamsLBM, programCtrl);

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
            rho.save<postProcess::LBMBin>(hostWriteBuffer, timeStep);

            U.save<postProcess::LBMBin>(hostWriteBuffer, timeStep);

            Pi.save<postProcess::LBMBin>(hostWriteBuffer, timeStep);

            // runTimeObjects.save(timeStep);
        }

        // Main kernel
        host::constexpr_for<0, NStreams()>(
            [&](const auto stream)
            {
                kernel::momentBasedLBM<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size<VelocitySet>(), streamsLBM.streams()[stream]>>>(
                    devPtrs,
                    blockHalo.readBuffer(VirtualDeviceIndex()),
                    blockHalo.writeBuffer(VirtualDeviceIndex()));
            });

        // Calculate S kernel
        // runTimeObjects.calculate();

        // Halo pointer swap
        blockHalo.swap(VirtualDeviceIndex());
    }

    return 0;
}

#endif