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
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Implementation of the multiphase moment representation with the D3Q27
    velocity set for hydrodynamics and D3Q7 for phase field evolution

Namespace
    LBM

SourceFiles
    phaseFieldD3Q27.cu

\*---------------------------------------------------------------------------*/

#include "phaseFieldD3Q27.cuh"

using namespace LBM;

__host__ [[nodiscard]] inline consteval label_t NStreams() noexcept { return 1; }

constexpr const device::label_t VirtualDeviceIndex = 0;

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

    // Phase field arrays
    device::array<field::FULL_FIELD, scalar_t, PhaseVelocitySet, time::instantaneous> phi("phi", mesh, programCtrl);

    const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs(
        rho.ptr(VirtualDeviceIndex),
        u.ptr(VirtualDeviceIndex),
        v.ptr(VirtualDeviceIndex),
        w.ptr(VirtualDeviceIndex),
        mxx.ptr(VirtualDeviceIndex),
        mxy.ptr(VirtualDeviceIndex),
        mxz.ptr(VirtualDeviceIndex),
        myy.ptr(VirtualDeviceIndex),
        myz.ptr(VirtualDeviceIndex),
        mzz.ptr(VirtualDeviceIndex),
        phi.ptr(VirtualDeviceIndex));

    const device::ptrCollection<NUMBER_MOMENTS<false>(), scalar_t> hydroPtrs(
        rho.ptr(VirtualDeviceIndex),
        u.ptr(VirtualDeviceIndex),
        v.ptr(VirtualDeviceIndex),
        w.ptr(VirtualDeviceIndex),
        mxx.ptr(VirtualDeviceIndex),
        mxy.ptr(VirtualDeviceIndex),
        mxz.ptr(VirtualDeviceIndex),
        myy.ptr(VirtualDeviceIndex),
        myz.ptr(VirtualDeviceIndex),
        mzz.ptr(VirtualDeviceIndex));

    // Setup Streams
    const streamHandler streamsLBM(programCtrl);

    // Allocate a buffer of pinned memory on the host for writing
    host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> hostWriteBuffer(mesh.size() * NUMBER_MOMENTS<true>(), mesh);

    objectRegistry<VelocitySet> runTimeObjects(hostWriteBuffer, mesh, rho, u, v, w, mxx, mxy, mxz, myy, myz, mzz, streamsLBM, programCtrl);

    device::haloSingle<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()> fBlockHalo(mesh, programCtrl);      // Hydrodynamic halo
    device::haloSingle<PhaseVelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()> gBlockHalo(mesh, programCtrl); // Phase field halo

    programCtrl.configure<smem_alloc_size<VelocitySet>()>(phaseFieldStream);

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
            {
                hostWriteBuffer.copy_from_device(
                    device::ptrCollection<11, scalar_t>{
                        rho.ptr(VirtualDeviceIndex),
                        u.ptr(VirtualDeviceIndex),
                        v.ptr(VirtualDeviceIndex),
                        w.ptr(VirtualDeviceIndex),
                        mxx.ptr(VirtualDeviceIndex),
                        mxy.ptr(VirtualDeviceIndex),
                        mxz.ptr(VirtualDeviceIndex),
                        myy.ptr(VirtualDeviceIndex),
                        myz.ptr(VirtualDeviceIndex),
                        mzz.ptr(VirtualDeviceIndex),
                        phi.ptr(VirtualDeviceIndex)},
                    mesh,
                    VirtualDeviceIndex);
            }

            fileIO::writeFile<time::instantaneous>(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                functionObjects::solutionVariableNames(true),
                hostWriteBuffer.data(),
                timeStep,
                rho.meanCount());

            runTimeObjects.save(timeStep);
        }

        // Main kernels
        host::constexpr_for<0, NStreams()>(
            [&](const auto stream)
            {
                phaseFieldStream<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size<VelocitySet>(), streamsLBM.streams()[stream]>>>(
                    devPtrs,
                    fBlockHalo.buffer(VirtualDeviceIndex),
                    gBlockHalo.buffer(VirtualDeviceIndex));

                phaseFieldCollide<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[stream]>>>(
                    devPtrs,
                    fBlockHalo.buffer(),
                    gBlockHalo.buffer());
            });

        // Calculate S kernel
        runTimeObjects.calculate(timeStep);
    }

    return 0;
}