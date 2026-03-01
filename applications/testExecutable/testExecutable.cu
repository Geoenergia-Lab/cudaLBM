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

// constexpr const label_t nx = 128;
// constexpr const label_t ny = 128;
// constexpr const label_t face_size = nx * ny;
// constexpr const label_t size = face_size * VelocitySet::QF();

// constexpr const label_t a = 0;
// constexpr const label_t b = 1;

launchBoundsD3Q27 __global__ void testKernel(const label_t *const ptrRestrict A, const label_t *const ptrRestrict B)
{
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0))
    {
        const label_t a = A[0];
        const label_t b = B[0];
        printf("A = %lu, B = %lu\n", static_cast<uint64_t>(a), static_cast<uint64_t>(b));
        return;
    }
}

template <const axis::type alpha, const int coeff>
__host__ static void handleGhostCells(
    std::vector<scalar_t> &face,
    const thread::array<scalar_t, VelocitySet::Q()> &pop,
    const label_t tx, const label_t ty, const label_t tz,
    const label_t bx, const label_t by, const label_t bz,
    const host::latticeMesh &mesh) noexcept
{
    static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::haloFace::handleGhostCells, "Potential issue with indexing: idxPop needs to be adapted to multi GPU."));

    axis::assertions::validate<alpha, axis::NOT_NULL>();

    velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

    constexpr const thread::array<label_t, VelocitySet::QF()> indices = velocitySet::template indices_on_face<VelocitySet, alpha, coeff>();

    const blockLabel_t Tx(tx, ty, tz);
    const blockLabel_t Bx(bx, by, bz);

    const label_t nxGPUs = mesh.nDevices<axis::X>();
    const label_t nyGPUs = mesh.nDevices<axis::Y>();

    for (label_t i = 0; i < VelocitySet::QF(); i++)
    {
        face[host::idxPop<alpha, VelocitySet::QF()>(i, Tx, Bx, mesh.nBlocks<axis::X>() / nxGPUs, mesh.nBlocks<axis::Y>() / nyGPUs)] = pop[indices[i]];
    }
}

template <const axis::type alpha, const int coeff>
__host__ [[nodiscard]] const std::vector<scalar_t> pop_init(
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &rho,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &u,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &v,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &w,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xx,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xy,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_xz,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_yy,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_yz,
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> &m_zz,
    const host::latticeMesh &mesh) noexcept
{
    static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::haloFace::initialise_pop, "Need to sort out indexing and decomposition between devices"));

    axis::assertions::validate<alpha, axis::NOT_NULL>();

    velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

    std::vector<scalar_t> face(mesh.nFacesPerDevice<alpha, VelocitySet::QF()>(), 0);

    // I think it is correct for this loop to be global
    // Because the initial conditions and file that we read from are already GPU-ordered
    host::forAll(
        mesh.nBlocks(),
        [&](const label_t bx, const label_t by, const label_t bz,
            const label_t tx, const label_t ty, const label_t tz)
        {
            const label_t base = host::idx(tx, ty, tz, bx, by, bz, mesh);

            // Handle ghost cells
            handleGhostCells<alpha, coeff>(
                face,
                VelocitySet::reconstruct(
                    {rho[base] + rho0(),
                     u[base],
                     v[base],
                     w[base],
                     m_xx[base],
                     m_xy[base],
                     m_xz[base],
                     m_yy[base],
                     m_yz[base],
                     m_zz[base]}),
                tx, ty, tz,
                bx, by, bz,
                mesh);
        });

    return face;
}

// __device__ scalar_t *EastHalo;

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    // Setup Streams
    const streamHandler streamsLBM(programCtrl);

    // Allocate host fields
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> rhoHost("rho", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> uHost("u", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> vHost("v", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> wHost("w", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> mxxHost("m_xx", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> mxyHost("m_xy", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> mxzHost("m_xz", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> myyHost("m_yy", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> myzHost("m_yz", mesh, programCtrl);
    const host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous> mzzHost("m_zz", mesh, programCtrl);

    BlockHalo blockHalo(mesh, programCtrl);

    // Initialise the East halo
    // We should allocate a fraction of this on the East device so that we can pull from it
    // const std::vector<scalar_t> pop = pop_init<axis::X, -1>(
    //     rhoHost,
    //     uHost, vHost, wHost,
    //     mxxHost, mxyHost, mxzHost, myyHost, myzHost, mzzHost,
    //     mesh);

    // const label_t nxBlocks = mesh.blocksPerDevice<axis::X>();
    // const label_t nyBlocks = mesh.blocksPerDevice<axis::Y>();
    // const label_t nzBlocks = mesh.blocksPerDevice<axis::Z>();

    // const label_t HaloAllocSize = nxBlocks * nyBlocks * block::nx() * block::ny() * VelocitySet::QF();

    // std::vector<scalar_t> pop_East(HaloAllocSize, 0);

    // for (label_t by = 0; by < nyBlocks; by++)
    // {
    //     for (label_t bx = 0; bx < nxBlocks; bx++)
    //     {
    //         for (label_t ty = 0; ty < block::ny(); ty++)
    //         {
    //             for (label_t tx = 0; tx < block::nx(); tx++)
    //             {
    //                 // const label_t base = host::idx(tx, ty, block::nz() - 1, bx, by, nzBlocks, mesh);
    //                 const label_t base = host::idx(tx, ty, block::nz() - 1, bx, by, mesh.nBlocks<axis::Z>() - 1, mesh);

    //                 // const auto a = VelocitySet::reconstruct(
    //                 //     {rhoHost[base] + rho0(),
    //                 //      uHost[base],
    //                 //      vHost[base],
    //                 //      wHost[base],
    //                 //      mxxHost[base],
    //                 //      mxyHost[base],
    //                 //      mxzHost[base],
    //                 //      myyHost[base],
    //                 //      myzHost[base],
    //                 //      mzzHost[base]});

    //                 //   std::cout << a[0] << std::endl;

    //                 handleGhostCells<axis::X, -1>(
    //                     pop_East,
    //                     VelocitySet::reconstruct(
    //                         {rhoHost[base] + rho0(),
    //                          uHost[base],
    //                          vHost[base],
    //                          wHost[base],
    //                          mxxHost[base],
    //                          mxyHost[base],
    //                          mxzHost[base],
    //                          myyHost[base],
    //                          myzHost[base],
    //                          mzzHost[base]}),
    //                     tx, ty, block::nz() - 1,
    //                     bx, by, mesh.nBlocks<axis::Z>() - 1,
    //                     mesh);
    //             }
    //         }
    //     }
    // }

    // // Now allocate East halo to West device
    // const scalar_t *EastHalo = device::allocate<scalar_t>(HaloAllocSize, programCtrl.deviceList()[0]);

    // const scalar_t *WestHalo = device::allocate<scalar_t>(HaloAllocSize, programCtrl.deviceList()[1]);

    // // Allocate the arrays on the device
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> rho(rhoHost, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> u("u", mesh, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> v("v", mesh, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> w("w", mesh, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxx("m_xx", mesh, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxy("m_xy", mesh, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxz("m_xz", mesh, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> myy("m_yy", mesh, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> myz("m_yz", mesh, programCtrl);
    // device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mzz("m_zz", mesh, programCtrl);

    // device::free(EastHalo, programCtrl.deviceList()[0]);
    // device::free(WestHalo, programCtrl.deviceList()[1]);

    // const label_t size = mesh.dimension<axis::X>() * mesh.dimension<axis::Y>() * VelocitySet::QF();

    // label_t *fGhost_0 = device::allocateArray(std::vector<label_t>(size, 0), programCtrl.deviceList()[0]);
    // label_t *gGhost_0 = device::allocateArray(std::vector<label_t>(size, 0), programCtrl.deviceList()[0]);

    // const label_t *fGhost_1 = device::allocateArray(std::vector<label_t>(size, 0), programCtrl.deviceList()[1]);
    // const label_t *gGhost_1 = device::allocateArray(std::vector<label_t>(size, 0), programCtrl.deviceList()[1]);

    // errorHandler::check(cudaFree(const_cast<label_t *>(fGhost_0)));
    // errorHandler::check(cudaFree(const_cast<label_t *>(gGhost_0)));
    // errorHandler::check(cudaFree(const_cast<label_t *>(fGhost_1)));
    // errorHandler::check(cudaFree(const_cast<label_t *>(gGhost_1)));

    // const std::vector<label_t> A(size, a);
    // const std::vector<label_t> B(size, b);

    // label_t *A0 = device::allocateArray(A, programCtrl.deviceList()[0]);
    // label_t *B0 = device::allocateArray(B, programCtrl.deviceList()[0]);

    // const label_t *A1 = device::allocateArray(A, programCtrl.deviceList()[1]);
    // const label_t *B1 = device::allocateArray(B, programCtrl.deviceList()[1]);

    // std::cout << std::endl;
    // std::cout << "Before cudaMemcpyPeer: " << std::endl;
    // std::cout << std::endl;

    // errorHandler::check(cudaSetDevice(programCtrl.deviceList()[0]));
    // std::cout << "GPU 0:" << std::endl;
    // testKernel<<<{1, 1, 1}, {8, 8, 8}, 0, 0>>>(A0, B0);
    // errorHandler::check(cudaDeviceSynchronize());

    // errorHandler::check(cudaSetDevice(programCtrl.deviceList()[1]));
    // std::cout << "GPU 1:" << std::endl;
    // testKernel<<<{1, 1, 1}, {8, 8, 8}, 0, 0>>>(A1, B1);
    // errorHandler::check(cudaDeviceSynchronize());

    // // Exchange memory: copy value of b to a
    // errorHandler::check(cudaMemcpyPeer(A0, 0, B1, 1, size * sizeof(label_t)));
    // errorHandler::check(cudaMemcpyPeer(B0, 0, A1, 1, size * sizeof(label_t)));

    // std::cout << "After cudaMemcpyPeer: " << std::endl;
    // std::cout << std::endl;

    // errorHandler::check(cudaSetDevice(programCtrl.deviceList()[0]));
    // std::cout << "GPU 0:" << std::endl;
    // testKernel<<<{1, 1, 1}, {8, 8, 8}, 0, 0>>>(A0, B0);
    // errorHandler::check(cudaDeviceSynchronize());

    // errorHandler::check(cudaSetDevice(programCtrl.deviceList()[1]));
    // std::cout << "GPU 1:" << std::endl;
    // testKernel<<<{1, 1, 1}, {8, 8, 8}, 0, 0>>>(A1, B1);
    // errorHandler::check(cudaDeviceSynchronize());

    // errorHandler::check(cudaFree(const_cast<label_t *>(A0)));
    // errorHandler::check(cudaFree(const_cast<label_t *>(A1)));
    // errorHandler::check(cudaFree(const_cast<label_t *>(B0)));
    // errorHandler::check(cudaFree(const_cast<label_t *>(B1)));

    // const host::latticeMesh mesh(programCtrl);

    // const device::array<field::FULL_FIELD, label_t, VelocitySet, time::instantaneous> d_A("A", mesh, 0, programCtrl);
    // const device::array<field::FULL_FIELD, label_t, VelocitySet, time::instantaneous> d_B("B", mesh, 0, programCtrl);

    // const host::latticeMesh mesh(programCtrl);

    // VelocitySet::print();

    // // const device::ptrCollection<10, scalar_t> devPtrs(
    // //     rho.ptr(VirtualDeviceIndex),
    // //     u.ptr(VirtualDeviceIndex),
    // //     v.ptr(VirtualDeviceIndex),
    // //     w.ptr(VirtualDeviceIndex),
    // //     mxx.ptr(VirtualDeviceIndex),
    // //     mxy.ptr(VirtualDeviceIndex),
    // //     mxz.ptr(VirtualDeviceIndex),
    // //     myy.ptr(VirtualDeviceIndex),
    // //     myz.ptr(VirtualDeviceIndex),
    // //     mzz.ptr(VirtualDeviceIndex));

    // // Setup Streams
    // const streamHandler streamsLBM(programCtrl);

    // // Allocate a buffer of pinned memory on the host for writing
    // host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> hostWriteBuffer(mesh.size() * NUMBER_MOMENTS(), mesh);

    // // objectRegistry<VelocitySet, NStreams()> runTimeObjects(hostWriteBuffer, mesh, devPtrs, streamsLBM, programCtrl);

    // BlockHalo blockHalo(mesh, programCtrl);

    // kernel::configure<smem_alloc_size()>(momentBasedD3Q27);

    // const runTimeIO IO(mesh, programCtrl);

    // // Temporarily allocate device halo pointers
    // // Should be allocated on GPU 0 and GPU 1
    // const label_t face_size = mesh.dimension<axis::X>(() * mesh.dimension<axis::Y>() * VelocitySet::QF();

    // scalar_t *const ptrRestrict ptr_0 = device::allocate(face_size, programCtrl.deviceList()[0]);
    // scalar_t *const ptrRestrict ptr_1 = device::allocate(face_size, programCtrl.deviceList()[1]);

    // for (label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    // {
    //     // Do the run-time IO
    //     if (programCtrl.print(timeStep))
    //     {
    //         std::cout << "Time: " << timeStep << std::endl;
    //     }

    //     // Checkpoint
    //     if (programCtrl.save(timeStep))
    //     {
    //         // Do this in a loop
    //         for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < programCtrl.deviceList().size(); VirtualDeviceIndex++)
    //         {
    //             hostWriteBuffer.copy_from_device(
    //                 device::ptrCollection<10, scalar_t>{
    //                     rho.ptr(VirtualDeviceIndex),
    //                     u.ptr(VirtualDeviceIndex),
    //                     v.ptr(VirtualDeviceIndex),
    //                     w.ptr(VirtualDeviceIndex),
    //                     mxx.ptr(VirtualDeviceIndex),
    //                     mxy.ptr(VirtualDeviceIndex),
    //                     mxz.ptr(VirtualDeviceIndex),
    //                     myy.ptr(VirtualDeviceIndex),
    //                     myz.ptr(VirtualDeviceIndex),
    //                     mzz.ptr(VirtualDeviceIndex)},
    //                 mesh,
    //                 VirtualDeviceIndex);
    //         }

    //         fileIO::writeFile<time::instantaneous>(
    //             programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
    //             mesh,
    //             functionObjects::solutionVariableNames,
    //             hostWriteBuffer.data(),
    //             timeStep,
    //             rho.meanCount());

    //         // runTimeObjects.save(timeStep);
    //     }

    //     // Main kernel
    //     for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < programCtrl.deviceList().size(); VirtualDeviceIndex++)
    //     {
    //         momentBasedD3Q27<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size(), streamsLBM.streams()[VirtualDeviceIndex]>>>(
    //             {rho.ptr(VirtualDeviceIndex),
    //              u.ptr(VirtualDeviceIndex),
    //              v.ptr(VirtualDeviceIndex),
    //              w.ptr(VirtualDeviceIndex),
    //              mxx.ptr(VirtualDeviceIndex),
    //              mxy.ptr(VirtualDeviceIndex),
    //              mxz.ptr(VirtualDeviceIndex),
    //              myy.ptr(VirtualDeviceIndex),
    //              myz.ptr(VirtualDeviceIndex),
    //              mzz.ptr(VirtualDeviceIndex)},
    //             blockHalo.fGhost(VirtualDeviceIndex),
    //             blockHalo.gGhost(VirtualDeviceIndex));
    //     }

    //     // Calculate S kernel
    //     // runTimeObjects.calculate(timeStep);

    //     // Do the memcpyPeerToPeer here
    //     {
    //         constexpr const label_t WestPointerIndex = 0;
    //         constexpr const label_t EastPointerIndex = 1;

    //         // I think it is correct to be copying West to East and East to West
    //         // scalar_t *const ptrRestrict ptr_0 = blockHalo.gGhost(0)[EastPointerIndex];
    //         // scalar_t *const ptrRestrict ptr_1 = blockHalo.gGhost(0)[WestPointerIndex];

    //         // First do the copy at the GPU's own boundaries
    //         errorHandler::check(cudaDeviceSynchronize());
    //         errorHandler::check(cudaSetDevice(programCtrl.deviceList()[0]));
    //         cudaMemcpyPeer(ptr_1, blockHalo.gGhost(0)[EastPointerIndex], face_size, cudaMemcpyDeviceToDevice);

    //         errorHandler::check(cudaDeviceSynchronize());
    //         errorHandler::check(cudaSetDevice(programCtrl.deviceList()[1]));
    //         cudaMemcpyPeer(ptr_0, blockHalo.gGhost(1)[EastPointerIndex], face_size, cudaMemcpyDeviceToDevice);
    //     }

    //     // Halo pointer swap
    //     for (label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < programCtrl.deviceList().size(); VirtualDeviceIndex++)
    //     {
    //         blockHalo.swap(VirtualDeviceIndex);
    //     }
    // }

    return 0;
}
