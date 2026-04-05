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
    A class handling the device halo. This class is used to exchange the
    microscopic velocity components at the edge of a CUDA block

Namespace
    LBM::device

SourceFiles
    halo.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALO_CUH
#define __MBLBM_HALO_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @class halo
         * @brief Manages halo regions for inter-block communication in CUDA LBM simulations
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         *
         * This class handles the exchange of distribution functions between adjacent
         * CUDA blocks during LBM simulations. It maintains double-buffered halo regions
         * to support efficient ping-pong swapping between computation steps.
         **/
        template <class VelocitySet, const bool x_periodic, const bool y_periodic, const bool z_periodic>
        class halo
        {
        public:
            /**
             * @brief Constructs halo regions from moment data and mesh
             * @param[in] mesh The lattice mesh
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] halo(const host::latticeMesh &mesh, const programControl &programCtrl) noexcept
                : readBuffer_(
                      [](const host::latticeMesh &m, const programControl &p)
                      {
                          if constexpr (VelocitySet::Q() == 7)
                          {
                              return makePhaseHalo(m, p);
                          }
                          else
                          {
                              return makeHydroHalo(m, p);
                          }
                      }(mesh, programCtrl)),
                  writeBuffer_(
                      [](const host::latticeMesh &m, const programControl &p)
                      {
                          if constexpr (VelocitySet::Q() == 7)
                          {
                              return makePhaseHalo(m, p);
                          }
                          else
                          {
                              return makeHydroHalo(m, p);
                          }
                      }(mesh, programCtrl)) {}

            /**
             * @brief Default destructor
             **/
            __host__ ~halo() {}

            /**
             * @brief Swaps read and write halo buffers
             * @note Synchronizes device before swapping to ensure all operations complete
             **/
            __host__ inline void swap(const host::label_t i) noexcept
            {
                errorHandler::checkInline(cudaDeviceSynchronize());
                swapPointers(i);
            }

            /**
             * @brief Swaps read and write halo buffers without device synchronization
             * @note Safe when producer/consumer kernels are stream-ordered
             **/
            __host__ inline void swapNoSync(const host::label_t i) noexcept
            {
                swapPointers(i);
            }

            /**
             * @brief Provides read-only access to the current read halo
             * @return Collection of const pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, const scalar_t> readBuffer(const host::label_t i) const noexcept
            {
                return {readBuffer_.x0Const(i), readBuffer_.x1Const(i), readBuffer_.y0Const(i), readBuffer_.y1Const(i), readBuffer_.z0Const(i), readBuffer_.z1Const(i)};
            }

            /**
             * @brief Provides mutable access to the current write halo
             * @return Collection of mutable pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, scalar_t> writeBuffer(const host::label_t i) noexcept
            {
                return {writeBuffer_.x0(i), writeBuffer_.x1(i), writeBuffer_.y0(i), writeBuffer_.y1(i), writeBuffer_.z0(i), writeBuffer_.z1(i)};
            }

            /**
             * @brief Provides read-only access to the current write halo
             * @return Collection of const pointers to write halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, const scalar_t> writeBufferConst(const host::label_t i) const noexcept
            {
                return {writeBuffer_.x0Const(i), writeBuffer_.x1Const(i), writeBuffer_.y0Const(i), writeBuffer_.y1Const(i), writeBuffer_.z0Const(i), writeBuffer_.z1Const(i)};
            }

            /**
             * @brief Compile-time periodicity query for X direction
             **/
            __device__ __host__ [[nodiscard]] static inline consteval bool periodicX() noexcept { return x_periodic; }

            /**
             * @brief Compile-time periodicity query for Y direction
             **/
            __device__ __host__ [[nodiscard]] static inline consteval bool periodicY() noexcept { return y_periodic; }

            /**
             * @brief Compile-time periodicity query for Z direction
             **/
            __device__ __host__ [[nodiscard]] static inline consteval bool periodicZ() noexcept { return z_periodic; }

            /**
             * @brief Loads halo population data from neighboring blocks
             * @param[out] pop Array to store loaded population values
             * @param[in] readBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            __device__ static inline constexpr void pull(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, const scalar_t> &readBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::halo::pull, "Potential issue with condition checking (e.g. West, East, etc)."));

                pull_direction<axis::X, x_periodic>(pop, readBuffer, Tx, Bx, point);

                pull_direction<axis::Y, y_periodic>(pop, readBuffer, Tx, Bx, point);

                pull_direction<axis::Z, z_periodic>(pop, readBuffer, Tx, Bx, point);
            }

            /**
             * @brief Loads a scalar value from either local memory or packed scalar halo faces
             * @tparam dx X-offset of the requested neighbor (-2..+2)
             * @tparam dy Y-offset of the requested neighbor (-2..+2)
             * @tparam dz Z-offset of the requested neighbor (-2..+2)
             * @param[in] scalarField Local scalar field array
             * @param[in] readBuffer Collection of packed scalar halo-face pointers (x0, x1, y0, y1, z0, z1)
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             * @param[in] point The global point coordinate
             **/
            template <const int dx, const int dy, const int dz>
            __device__ static inline constexpr scalar_t pull_scalar(
                const scalar_t *const ptrRestrict scalarField,
                const device::ptrCollection<6, const scalar_t> &readBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                constexpr int scalarHaloDepth = 2;

                static_assert((dx >= -scalarHaloDepth) && (dx <= scalarHaloDepth), "dx must be in [-2, 2].");
                static_assert((dy >= -scalarHaloDepth) && (dy <= scalarHaloDepth), "dy must be in [-2, 2].");
                static_assert((dz >= -scalarHaloDepth) && (dz <= scalarHaloDepth), "dz must be in [-2, 2].");

                if constexpr (dx == 0 && dy == 0 && dz == 0)
                {
                    return __ldg(&(scalarField[device::idx(Tx, Bx)]));
                }

                const int nX = static_cast<int>(block::n<axis::X>());
                const int nY = static_cast<int>(block::n<axis::Y>());
                const int nZ = static_cast<int>(block::n<axis::Z>());

                const int shiftedX = static_cast<int>(Tx.value<axis::X>()) + dx;
                const int shiftedY = static_cast<int>(Tx.value<axis::Y>()) + dy;
                const int shiftedZ = static_cast<int>(Tx.value<axis::Z>()) + dz;

                if constexpr (!x_periodic)
                {
                    const int gx = static_cast<int>(point.value<axis::X>()) + dx;
                    if ((gx < 0) || (gx >= static_cast<int>(device::n<axis::X>())))
                    {
                        return static_cast<scalar_t>(0);
                    }
                }

                if constexpr (!y_periodic)
                {
                    const int gy = static_cast<int>(point.value<axis::Y>()) + dy;
                    if ((gy < 0) || (gy >= static_cast<int>(device::n<axis::Y>())))
                    {
                        return static_cast<scalar_t>(0);
                    }
                }

                if constexpr (!z_periodic)
                {
                    const int gz = static_cast<int>(point.value<axis::Z>()) + dz;
                    if ((gz < 0) || (gz >= static_cast<int>(device::n<axis::Z>())))
                    {
                        return static_cast<scalar_t>(0);
                    }
                }

                const bool crossMinusX = shiftedX < 0;
                const bool crossMinusY = shiftedY < 0;
                const bool crossMinusZ = shiftedZ < 0;

                const bool crossPlusX = shiftedX >= nX;
                const bool crossPlusY = shiftedY >= nY;
                const bool crossPlusZ = shiftedZ >= nZ;

                const device::label_t tx = static_cast<device::label_t>(crossMinusX ? (shiftedX + nX) : (crossPlusX ? (shiftedX - nX) : shiftedX));
                const device::label_t ty = static_cast<device::label_t>(crossMinusY ? (shiftedY + nY) : (crossPlusY ? (shiftedY - nY) : shiftedY));
                const device::label_t tz = static_cast<device::label_t>(crossMinusZ ? (shiftedZ + nZ) : (crossPlusZ ? (shiftedZ - nZ) : shiftedZ));

                const bool haloX = crossMinusX || crossPlusX;
                const bool haloY = crossMinusY || crossPlusY;
                const bool haloZ = crossMinusZ || crossPlusZ;

                const device::label_t bx =
                    crossMinusX ? Bx.shifted_block<axis::X, -1>() : (crossPlusX ? Bx.shifted_block<axis::X, +1>() : Bx.value<axis::X>());
                const device::label_t by =
                    crossMinusY ? Bx.shifted_block<axis::Y, -1>() : (crossPlusY ? Bx.shifted_block<axis::Y, +1>() : Bx.value<axis::Y>());
                const device::label_t bz =
                    crossMinusZ ? Bx.shifted_block<axis::Z, -1>() : (crossPlusZ ? Bx.shifted_block<axis::Z, +1>() : Bx.value<axis::Z>());

                int pointShiftX = static_cast<int>(point.value<axis::X>()) + dx;
                int pointShiftY = static_cast<int>(point.value<axis::Y>()) + dy;
                int pointShiftZ = static_cast<int>(point.value<axis::Z>()) + dz;

                if constexpr (x_periodic)
                {
                    if (pointShiftX < 0)
                    {
                        pointShiftX += static_cast<int>(device::n<axis::X>());
                    }
                    else if (pointShiftX >= static_cast<int>(device::n<axis::X>()))
                    {
                        pointShiftX -= static_cast<int>(device::n<axis::X>());
                    }
                }

                if constexpr (y_periodic)
                {
                    if (pointShiftY < 0)
                    {
                        pointShiftY += static_cast<int>(device::n<axis::Y>());
                    }
                    else if (pointShiftY >= static_cast<int>(device::n<axis::Y>()))
                    {
                        pointShiftY -= static_cast<int>(device::n<axis::Y>());
                    }
                }

                if constexpr (z_periodic)
                {
                    if (pointShiftZ < 0)
                    {
                        pointShiftZ += static_cast<int>(device::n<axis::Z>());
                    }
                    else if (pointShiftZ >= static_cast<int>(device::n<axis::Z>()))
                    {
                        pointShiftZ -= static_cast<int>(device::n<axis::Z>());
                    }
                }

#ifndef FORCE_MULTI_GPU_SCALAR_HALO_TEST
                const int localSizeX = static_cast<int>(block::n<axis::X>() * device::NUM_BLOCK<axis::X>());
                const int localSizeY = static_cast<int>(block::n<axis::Y>() * device::NUM_BLOCK<axis::Y>());
                const int localSizeZ = static_cast<int>(block::n<axis::Z>() * device::NUM_BLOCK<axis::Z>());

                const int localStartX = static_cast<int>(block::n<axis::X>() * device::BLOCK_OFFSET_X);
                const int localStartY = static_cast<int>(block::n<axis::Y>() * device::BLOCK_OFFSET_Y);
                const int localStartZ = static_cast<int>(block::n<axis::Z>() * device::BLOCK_OFFSET_Z);

                const bool withinLocalSubdomain =
                    (pointShiftX >= localStartX) && (pointShiftX < (localStartX + localSizeX)) &&
                    (pointShiftY >= localStartY) && (pointShiftY < (localStartY + localSizeY)) &&
                    (pointShiftZ >= localStartZ) && (pointShiftZ < (localStartZ + localSizeZ));

                if (withinLocalSubdomain)
                {
                    return __ldg(&(scalarField[device::idx(tx, ty, tz, bx, by, bz)]));
                }
#endif

                if (!(haloX || haloY || haloZ))
                {
                    return __ldg(&(scalarField[device::idx(tx, ty, tz, bx, by, bz)]));
                }

                if (haloX)
                {
                    const device::label_t layer = static_cast<device::label_t>(crossMinusX ? (-shiftedX - 1) : (shiftedX - nX));
                    const device::label_t faceIdx =
                        (layer == static_cast<device::label_t>(0))
                            ? idxPop<axis::X, 0, scalarHaloDepth>(ty, tz, bx, by, bz)
                            : idxPop<axis::X, 1, scalarHaloDepth>(ty, tz, bx, by, bz);

                    if (crossMinusX)
                    {
                        return __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::X, +1>())>()[faceIdx]));
                    }

                    return __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::X, -1>())>()[faceIdx]));
                }

                if (haloY)
                {
                    const device::label_t layer = static_cast<device::label_t>(crossMinusY ? (-shiftedY - 1) : (shiftedY - nY));
                    const device::label_t faceIdx =
                        (layer == static_cast<device::label_t>(0))
                            ? idxPop<axis::Y, 0, scalarHaloDepth>(tx, tz, bx, by, bz)
                            : idxPop<axis::Y, 1, scalarHaloDepth>(tx, tz, bx, by, bz);

                    if (crossMinusY)
                    {
                        return __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Y, +1>())>()[faceIdx]));
                    }

                    return __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Y, -1>())>()[faceIdx]));
                }

                const device::label_t layer = static_cast<device::label_t>(crossMinusZ ? (-shiftedZ - 1) : (shiftedZ - nZ));
                const device::label_t faceIdx =
                    (layer == static_cast<device::label_t>(0))
                        ? idxPop<axis::Z, 0, scalarHaloDepth>(tx, ty, bx, by, bz)
                        : idxPop<axis::Z, 1, scalarHaloDepth>(tx, ty, bx, by, bz);

                if (crossMinusZ)
                {
                    return __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Z, +1>())>()[faceIdx]));
                }

                return __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Z, -1>())>()[faceIdx]));
            }

            /**
             * @brief Loads a scalar value without halo-face buffers (single-GPU path)
             * @tparam dx X-offset of the requested neighbor (-2..+2)
             * @tparam dy Y-offset of the requested neighbor (-2..+2)
             * @tparam dz Z-offset of the requested neighbor (-2..+2)
             **/
            template <const int dx, const int dy, const int dz>
            __device__ static inline constexpr scalar_t pull_scalar_local(
                const scalar_t *const ptrRestrict scalarField,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                constexpr int scalarHaloDepth = 2;

                static_assert((dx >= -scalarHaloDepth) && (dx <= scalarHaloDepth), "dx must be in [-2, 2].");
                static_assert((dy >= -scalarHaloDepth) && (dy <= scalarHaloDepth), "dy must be in [-2, 2].");
                static_assert((dz >= -scalarHaloDepth) && (dz <= scalarHaloDepth), "dz must be in [-2, 2].");

                if constexpr (dx == 0 && dy == 0 && dz == 0)
                {
                    return __ldg(&(scalarField[device::idx(Tx, Bx)]));
                }

                const int nX = static_cast<int>(block::n<axis::X>());
                const int nY = static_cast<int>(block::n<axis::Y>());
                const int nZ = static_cast<int>(block::n<axis::Z>());

                const int shiftedX = static_cast<int>(Tx.value<axis::X>()) + dx;
                const int shiftedY = static_cast<int>(Tx.value<axis::Y>()) + dy;
                const int shiftedZ = static_cast<int>(Tx.value<axis::Z>()) + dz;

                if constexpr (!x_periodic)
                {
                    const int gx = static_cast<int>(point.value<axis::X>()) + dx;
                    if ((gx < 0) || (gx >= static_cast<int>(device::n<axis::X>())))
                    {
                        return static_cast<scalar_t>(0);
                    }
                }

                if constexpr (!y_periodic)
                {
                    const int gy = static_cast<int>(point.value<axis::Y>()) + dy;
                    if ((gy < 0) || (gy >= static_cast<int>(device::n<axis::Y>())))
                    {
                        return static_cast<scalar_t>(0);
                    }
                }

                if constexpr (!z_periodic)
                {
                    const int gz = static_cast<int>(point.value<axis::Z>()) + dz;
                    if ((gz < 0) || (gz >= static_cast<int>(device::n<axis::Z>())))
                    {
                        return static_cast<scalar_t>(0);
                    }
                }

                const bool crossMinusX = shiftedX < 0;
                const bool crossMinusY = shiftedY < 0;
                const bool crossMinusZ = shiftedZ < 0;

                const bool crossPlusX = shiftedX >= nX;
                const bool crossPlusY = shiftedY >= nY;
                const bool crossPlusZ = shiftedZ >= nZ;

                const device::label_t tx = static_cast<device::label_t>(crossMinusX ? (shiftedX + nX) : (crossPlusX ? (shiftedX - nX) : shiftedX));
                const device::label_t ty = static_cast<device::label_t>(crossMinusY ? (shiftedY + nY) : (crossPlusY ? (shiftedY - nY) : shiftedY));
                const device::label_t tz = static_cast<device::label_t>(crossMinusZ ? (shiftedZ + nZ) : (crossPlusZ ? (shiftedZ - nZ) : shiftedZ));

                const device::label_t bx =
                    crossMinusX ? Bx.shifted_block<axis::X, -1>() : (crossPlusX ? Bx.shifted_block<axis::X, +1>() : Bx.value<axis::X>());
                const device::label_t by =
                    crossMinusY ? Bx.shifted_block<axis::Y, -1>() : (crossPlusY ? Bx.shifted_block<axis::Y, +1>() : Bx.value<axis::Y>());
                const device::label_t bz =
                    crossMinusZ ? Bx.shifted_block<axis::Z, -1>() : (crossPlusZ ? Bx.shifted_block<axis::Z, +1>() : Bx.value<axis::Z>());

                return __ldg(&(scalarField[device::idx(tx, ty, tz, bx, by, bz)]));
            }

            /**
             * @brief Saves a scalar value to packed two-layer scalar halo faces
             * @param[in] value Scalar value to save in halo faces
             * @param[in] writeBuffer Collection of packed scalar halo-face pointers (x0, x1, y0, y1, z0, z1)
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             * @param[in] point The global point coordinate
             **/
            __device__ static inline constexpr void save_scalar(
                const scalar_t value,
                const device::ptrCollection<6, scalar_t> &writeBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                constexpr device::label_t scalarHaloDepth = static_cast<device::label_t>(2);

#ifndef FORCE_MULTI_GPU_SCALAR_HALO_TEST
                const device::label_t localSizeX = block::n<axis::X>() * device::NUM_BLOCK<axis::X>();
                const device::label_t localSizeY = block::n<axis::Y>() * device::NUM_BLOCK<axis::Y>();
                const device::label_t localSizeZ = block::n<axis::Z>() * device::NUM_BLOCK<axis::Z>();

                if ((localSizeX == device::n<axis::X>()) &&
                    (localSizeY == device::n<axis::Y>()) &&
                    (localSizeZ == device::n<axis::Z>()))
                {
                    return;
                }
#endif

                if ((Tx.value<axis::X>() == static_cast<device::label_t>(0)) &&
                    (x_periodic || (point.value<axis::X>() > static_cast<device::label_t>(0))))
                {
                    writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::X, -1>())>()[idxPop<axis::X, 0, scalarHaloDepth>(Tx.value<axis::Y>(), Tx.value<axis::Z>(), Bx)] = value;
                }

                if constexpr (block::n<axis::X>() > static_cast<device::label_t>(1))
                {
                    if ((Tx.value<axis::X>() == static_cast<device::label_t>(1)) &&
                        (x_periodic || (point.value<axis::X>() > static_cast<device::label_t>(1))))
                    {
                        writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::X, -1>())>()[idxPop<axis::X, 1, scalarHaloDepth>(Tx.value<axis::Y>(), Tx.value<axis::Z>(), Bx)] = value;
                    }
                }

                if ((Tx.value<axis::X>() == (block::n<axis::X>() - static_cast<device::label_t>(1))) &&
                    (x_periodic || (point.value<axis::X>() < (device::n<axis::X>() - static_cast<device::label_t>(1)))))
                {
                    writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::X, +1>())>()[idxPop<axis::X, 0, scalarHaloDepth>(Tx.value<axis::Y>(), Tx.value<axis::Z>(), Bx)] = value;
                }

                if constexpr (block::n<axis::X>() > static_cast<device::label_t>(1))
                {
                    if ((Tx.value<axis::X>() == (block::n<axis::X>() - static_cast<device::label_t>(2))) &&
                        (x_periodic || (point.value<axis::X>() < (device::n<axis::X>() - static_cast<device::label_t>(2)))))
                    {
                        writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::X, +1>())>()[idxPop<axis::X, 1, scalarHaloDepth>(Tx.value<axis::Y>(), Tx.value<axis::Z>(), Bx)] = value;
                    }
                }

                if ((Tx.value<axis::Y>() == static_cast<device::label_t>(0)) &&
                    (y_periodic || (point.value<axis::Y>() > static_cast<device::label_t>(0))))
                {
                    writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Y, -1>())>()[idxPop<axis::Y, 0, scalarHaloDepth>(Tx.value<axis::X>(), Tx.value<axis::Z>(), Bx)] = value;
                }

                if constexpr (block::n<axis::Y>() > static_cast<device::label_t>(1))
                {
                    if ((Tx.value<axis::Y>() == static_cast<device::label_t>(1)) &&
                        (y_periodic || (point.value<axis::Y>() > static_cast<device::label_t>(1))))
                    {
                        writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Y, -1>())>()[idxPop<axis::Y, 1, scalarHaloDepth>(Tx.value<axis::X>(), Tx.value<axis::Z>(), Bx)] = value;
                    }
                }

                if ((Tx.value<axis::Y>() == (block::n<axis::Y>() - static_cast<device::label_t>(1))) &&
                    (y_periodic || (point.value<axis::Y>() < (device::n<axis::Y>() - static_cast<device::label_t>(1)))))
                {
                    writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Y, +1>())>()[idxPop<axis::Y, 0, scalarHaloDepth>(Tx.value<axis::X>(), Tx.value<axis::Z>(), Bx)] = value;
                }

                if constexpr (block::n<axis::Y>() > static_cast<device::label_t>(1))
                {
                    if ((Tx.value<axis::Y>() == (block::n<axis::Y>() - static_cast<device::label_t>(2))) &&
                        (y_periodic || (point.value<axis::Y>() < (device::n<axis::Y>() - static_cast<device::label_t>(2)))))
                    {
                        writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Y, +1>())>()[idxPop<axis::Y, 1, scalarHaloDepth>(Tx.value<axis::X>(), Tx.value<axis::Z>(), Bx)] = value;
                    }
                }

                if ((Tx.value<axis::Z>() == static_cast<device::label_t>(0)) &&
                    (z_periodic || (point.value<axis::Z>() > static_cast<device::label_t>(0))))
                {
                    writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Z, -1>())>()[idxPop<axis::Z, 0, scalarHaloDepth>(Tx.value<axis::X>(), Tx.value<axis::Y>(), Bx)] = value;
                }

                if constexpr (block::n<axis::Z>() > static_cast<device::label_t>(1))
                {
                    if ((Tx.value<axis::Z>() == static_cast<device::label_t>(1)) &&
                        (z_periodic || (point.value<axis::Z>() > static_cast<device::label_t>(1))))
                    {
                        writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Z, -1>())>()[idxPop<axis::Z, 1, scalarHaloDepth>(Tx.value<axis::X>(), Tx.value<axis::Y>(), Bx)] = value;
                    }
                }

                if ((Tx.value<axis::Z>() == (block::n<axis::Z>() - static_cast<device::label_t>(1))) &&
                    (z_periodic || (point.value<axis::Z>() < (device::n<axis::Z>() - static_cast<device::label_t>(1)))))
                {
                    writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Z, +1>())>()[idxPop<axis::Z, 0, scalarHaloDepth>(Tx.value<axis::X>(), Tx.value<axis::Y>(), Bx)] = value;
                }

                if constexpr (block::n<axis::Z>() > static_cast<device::label_t>(1))
                {
                    if ((Tx.value<axis::Z>() == (block::n<axis::Z>() - static_cast<device::label_t>(2))) &&
                        (z_periodic || (point.value<axis::Z>() < (device::n<axis::Z>() - static_cast<device::label_t>(2)))))
                    {
                        writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<axis::Z, +1>())>()[idxPop<axis::Z, 1, scalarHaloDepth>(Tx.value<axis::X>(), Tx.value<axis::Y>(), Bx)] = value;
                    }
                }
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @param[in] pop Array containing population values to save
             * @param[out] writeBuffer Collection of pointers to the halo faces
             *
             * This device function saves population values to halo regions for
             * neighboring blocks to read.
             **/
            __device__ static inline constexpr void save(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const thread::array<scalar_t, 10> &moments,
                const device::ptrCollection<6, scalar_t> &writeBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::halo::save, "Potential issue with condition checking (e.g. West, East, etc)."));
                if constexpr (VelocitySet::Q() != 7)
                {
                    VelocitySet::template reconstruct<false>(pop, moments);
                }

                save_direction<axis::X, x_periodic>(pop, writeBuffer, Tx, Bx, point);

                save_direction<axis::Y, y_periodic>(pop, writeBuffer, Tx, Bx, point);

                save_direction<axis::Z, z_periodic>(pop, writeBuffer, Tx, Bx, point);
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @param[in] pop Array containing population values to save
             * @param[out] writeBuffer Collection of pointers to the halo faces
             *
             * This device function saves population values to halo regions for
             * neighboring blocks to read.
             **/
            __device__ static inline constexpr void save(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const thread::array<scalar_t, 11> &moments,
                const device::ptrCollection<6, scalar_t> &writeBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                static_assert(MULTI_GPU_ASSERTION(), MULTI_GPU_MSG_NOTE(device::halo::save, "Potential issue with condition checking (e.g. West, East, etc)."));
                if constexpr (VelocitySet::Q() != 7)
                {
                    VelocitySet::template reconstruct<false>(pop, moments);
                }

                save_direction<axis::X, x_periodic>(pop, writeBuffer, Tx, Bx, point);

                save_direction<axis::Y, y_periodic>(pop, writeBuffer, Tx, Bx, point);

                save_direction<axis::Z, z_periodic>(pop, writeBuffer, Tx, Bx, point);
            }

#include "haloSharedMemoryOperations.cuh"

        private:
            /**
             * @brief Swaps read/write face pointers
             **/
            __host__ inline void swapPointers(const host::label_t i) noexcept
            {
                std::swap(readBuffer_.x0Ref(i), writeBuffer_.x0Ref(i));
                std::swap(readBuffer_.x1Ref(i), writeBuffer_.x1Ref(i));
                std::swap(readBuffer_.y0Ref(i), writeBuffer_.y0Ref(i));
                std::swap(readBuffer_.y1Ref(i), writeBuffer_.y1Ref(i));
                std::swap(readBuffer_.z0Ref(i), writeBuffer_.z0Ref(i));
                std::swap(readBuffer_.z1Ref(i), writeBuffer_.z1Ref(i));
            }

            /**
             * @brief The individual halo objects
             **/
            haloFace<VelocitySet> readBuffer_;
            haloFace<VelocitySet> writeBuffer_;

            /**
             * @brief Construct halo regions from hydrodynamic moments only
             **/
            __host__ [[nodiscard]] static inline haloFace<VelocitySet>
            makeHydroHalo(const host::latticeMesh &mesh, const programControl &programCtrl) noexcept
            {
                return haloFace<VelocitySet>(
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("rho", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("u", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("v", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("w", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xx", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xy", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xz", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_yy", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_yz", mesh, programCtrl),
                    host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_zz", mesh, programCtrl),
                    mesh,
                    programCtrl);
            }

            /**
             * @brief Construct halo regions from hydrodynamic moments + phase field
             **/
            __host__ [[nodiscard]] static inline haloFace<VelocitySet>
            makePhaseHalo(const host::latticeMesh &mesh, const programControl &programCtrl) noexcept
            {
                auto rho = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("rho", mesh, programCtrl);
                auto u = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("u", mesh, programCtrl);
                auto v = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("v", mesh, programCtrl);
                auto w = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("w", mesh, programCtrl);
                auto m_xx = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xx", mesh, programCtrl);
                auto m_xy = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xy", mesh, programCtrl);
                auto m_xz = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_xz", mesh, programCtrl);
                auto m_yy = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_yy", mesh, programCtrl);
                auto m_yz = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_yz", mesh, programCtrl);
                auto m_zz = host::array<host::PAGED, scalar_t, VelocitySet, time::instantaneous>("m_zz", mesh, programCtrl);
                auto phi = host::array<host::PAGED, scalar_t, D3Q7, time::instantaneous>("phi", mesh, programCtrl);

                return haloFace<VelocitySet>(
                    rho,
                    u,
                    v,
                    w,
                    m_xx,
                    m_xy,
                    m_xz,
                    m_yy,
                    m_yz,
                    m_zz,
                    mesh,
                    programCtrl,
                    &phi);
            }

            /**
             * @brief Returns the streaming index for a given axis and velocity
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam v The velocity component (-1 or 1)
             * @param[i] i The index of the velocity
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] static inline consteval device::label_t streaming_index(const device::label_t i) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                return static_cast<device::label_t>(velocitySet::template indices_on_face<VelocitySet, alpha, coeff>()[i]);
            }

            /**
             * @brief Checks if the current thread is at a block boundary in a given direction, accounting for periodicity
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam coeff The normal direction; -1 for negative face, +1 for positive face
             * @tparam isPeriodic Whether the domain is periodic in this direction
             * @param[in] alpha_v The global coordinate in the alpha direction
             * @param[in] Tx Three-dimensional thread coordinates
             * @return True if the thread is at the specified block boundary and not at the domain edge (if non-periodic)
             **/
            template <const axis::type alpha, const int coeff, const bool isPeriodic>
            __device__ [[nodiscard]] static inline constexpr bool boundaryCheck(const device::label_t alpha_v, const thread::coordinate &Tx) noexcept
            {
                if constexpr (coeff == -1)
                {
                    if constexpr (isPeriodic)
                    {
                        return (Tx.value<alpha>() == 0);
                    }
                    else
                    {
                        return (Tx.value<alpha>() == 0 && alpha_v != 0);
                    }
                }

                if constexpr (coeff == 1)
                {
                    if constexpr (isPeriodic)
                    {
                        return (Tx.value<alpha>() == (block::n<alpha>() - 1));
                    }
                    else
                    {
                        return (Tx.value<alpha>() == (block::n<alpha>() - 1) && alpha_v != (device::n<alpha>() - 1));
                    }
                }
            }

            /**
             * @brief Selects between shifted or central thread coordinates based upon a coefficient
             * @tparam coeff The velocity set coefficient (-1, 0, +1)
             * @param[in] dt The left and right hand side shifted thread coordinates
             * @param[in] t The thread coordinate
             **/
            template <const int coeff>
            __device__ [[nodiscard]] static inline constexpr device::label_t thread_stencil(const thread::array<device::label_t, 2> &dt, const device::label_t t) noexcept
            {
                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::CAN_BE_NULL>();

                if constexpr (coeff == -1)
                {
                    return dt[0];
                }

                if constexpr (coeff == 0)
                {
                    return t;
                }

                if constexpr (coeff == 1)
                {
                    return dt[1];
                }
            }

            /**
             * @brief Selects between shifted or central block coordinates based upon a coefficient
             * @tparam alpha The axis (X, Y or Z)
             * @tparam coeff The velocity set coefficient (-1, 0, +1)
             * @param[in] t The thread coordinate
             * @param[in] b_shifted The shifted block
             * @param[in] b The current block
             **/
            template <const axis::type alpha, const int coeff>
            __device__ [[nodiscard]] static inline constexpr device::label_t block_stencil(const device::label_t t, const device::label_t b_shifted, const device::label_t b) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::CAN_BE_NULL>();

                if constexpr (coeff == -1)
                {
                    return (t == 0) ? (b_shifted) : (b);
                }

                if constexpr (coeff == 0)
                {
                    return b;
                }

                if constexpr (coeff == 1)
                {
                    return (t == block::n<alpha>() - 1) ? (b_shifted) : (b);
                }
            }

            /**
             * @brief Loads the populations from the halo into the pop array for a particular face
             * @tparam alpha The axis direction
             * @tparam PtrIndex The index of the pointer corresponding to the halo face
             * @tparam coeff The normal direction; -1 or +1
             * @param[out] pop Array to store loaded population values
             * @param[in] readBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            template <const axis::type alpha, const int coeff>
            __device__ static inline constexpr void pull_face(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, const scalar_t> &readBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                const thread::array<device::label_t, 2> dBx{Bx.shifted_block<axis::X, -1>(), Bx.shifted_block<axis::X, +1>()};
                const thread::array<device::label_t, 2> dBy{Bx.shifted_block<axis::Y, -1>(), Bx.shifted_block<axis::Y, +1>()};
                const thread::array<device::label_t, 2> dBz{Bx.shifted_block<axis::Z, -1>(), Bx.shifted_block<axis::Z, +1>()};

                const thread::array<device::label_t, 2> da{Tx.shifted_coordinate<axis::orthogonal<alpha, 0>(), -1>(), Tx.shifted_coordinate<axis::orthogonal<alpha, 0>(), +1>()};
                const thread::array<device::label_t, 2> db{Tx.shifted_coordinate<axis::orthogonal<alpha, 1>(), -1>(), Tx.shifted_coordinate<axis::orthogonal<alpha, 1>(), +1>()};

                if constexpr (alpha == axis::X)
                {
                    const device::label_t b_x = block_stencil<axis::X, -coeff>(
                        Tx.value<axis::X>(),
                        thread_stencil<-coeff>(dBx, Bx.value<axis::X>()),
                        Bx.value<axis::X>());

                    device::constexpr_for<0, VelocitySet::QF()>(
                        [&](const auto i)
                        {
                            const device::label_t t_a = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 0>()>()[streaming_index<alpha, coeff>(i)]>(da, Tx.value<axis::orthogonal<alpha, 0>()>());
                            const device::label_t t_b = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 1>()>()[streaming_index<alpha, coeff>(i)]>(db, Tx.value<axis::orthogonal<alpha, 1>()>());

                            // Then we should select the true block based on the thread
                            const device::label_t b_y = block_stencil<axis::Y, -VelocitySet::template c<int, axis::Y>()[streaming_index<alpha, coeff>(i)]>(
                                Tx.value<axis::Y>(),
                                thread_stencil<-VelocitySet::template c<int, axis::Y>()[streaming_index<alpha, coeff>(i)]>(dBy, Bx.value<axis::Y>()),
                                Bx.value<axis::Y>());
                            const device::label_t b_z = block_stencil<axis::Z, -VelocitySet::template c<int, axis::Z>()[streaming_index<alpha, coeff>(i)]>(
                                Tx.value<axis::Z>(),
                                thread_stencil<-VelocitySet::template c<int, axis::Z>()[streaming_index<alpha, coeff>(i)]>(dBz, Bx.value<axis::Z>()),
                                Bx.value<axis::Z>());

                            pop[q_i<streaming_index<alpha, coeff>(i)>()] = __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<alpha, coeff>())>()[idxPop<alpha, i, VelocitySet::QF()>(t_a, t_b, b_x, b_y, b_z)]));
                        });
                }

                if constexpr (alpha == axis::Y)
                {
                    const device::label_t b_y = block_stencil<axis::Y, -coeff>(
                        Tx.value<axis::Y>(),
                        thread_stencil<-coeff>(dBy, Bx.value<axis::Y>()),
                        Bx.value<axis::Y>());

                    device::constexpr_for<0, VelocitySet::QF()>(
                        [&](const auto i)
                        {
                            const device::label_t t_a = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 0>()>()[streaming_index<alpha, coeff>(i)]>(da, Tx.value<axis::orthogonal<alpha, 0>()>());
                            const device::label_t t_b = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 1>()>()[streaming_index<alpha, coeff>(i)]>(db, Tx.value<axis::orthogonal<alpha, 1>()>());

                            // Then we should select the true block based on the thread
                            const device::label_t b_x = block_stencil<axis::X, -VelocitySet::template c<int, axis::X>()[streaming_index<alpha, coeff>(i)]>(
                                Tx.value<axis::X>(),
                                thread_stencil<-VelocitySet::template c<int, axis::X>()[streaming_index<alpha, coeff>(i)]>(dBx, Bx.value<axis::X>()),
                                Bx.value<axis::X>());
                            const device::label_t b_z = block_stencil<axis::Z, -VelocitySet::template c<int, axis::Z>()[streaming_index<alpha, coeff>(i)]>(
                                Tx.value<axis::Z>(),
                                thread_stencil<-VelocitySet::template c<int, axis::Z>()[streaming_index<alpha, coeff>(i)]>(dBz, Bx.value<axis::Z>()),
                                Bx.value<axis::Z>());

                            pop[q_i<streaming_index<alpha, coeff>(i)>()] = __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<alpha, coeff>())>()[idxPop<alpha, i, VelocitySet::QF()>(t_a, t_b, b_x, b_y, b_z)]));
                        });
                }

                if constexpr (alpha == axis::Z)
                {
                    const device::label_t b_z = block_stencil<axis::Z, -coeff>(
                        Tx.value<axis::Z>(),
                        thread_stencil<-coeff>(dBz, Bx.value<axis::Z>()),
                        Bx.value<axis::Z>());

                    device::constexpr_for<0, VelocitySet::QF()>(
                        [&](const auto i)
                        {
                            const device::label_t t_a = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 0>()>()[streaming_index<alpha, coeff>(i)]>(da, Tx.value<axis::orthogonal<alpha, 0>()>());
                            const device::label_t t_b = thread_stencil<-VelocitySet::template c<int, axis::orthogonal<alpha, 1>()>()[streaming_index<alpha, coeff>(i)]>(db, Tx.value<axis::orthogonal<alpha, 1>()>());

                            // Then we should select the true block based on the thread
                            const device::label_t b_x = block_stencil<axis::X, -VelocitySet::template c<int, axis::X>()[streaming_index<alpha, coeff>(i)]>(
                                Tx.value<axis::X>(),
                                thread_stencil<-VelocitySet::template c<int, axis::X>()[streaming_index<alpha, coeff>(i)]>(dBx, Bx.value<axis::X>()),
                                Bx.value<axis::X>());
                            const device::label_t b_y = block_stencil<axis::Y, -VelocitySet::template c<int, axis::Y>()[streaming_index<alpha, coeff>(i)]>(
                                Tx.value<axis::Y>(),
                                thread_stencil<-VelocitySet::template c<int, axis::Y>()[streaming_index<alpha, coeff>(i)]>(dBy, Bx.value<axis::Y>()),
                                Bx.value<axis::Y>());

                            pop[q_i<streaming_index<alpha, coeff>(i)>()] = __ldg(&(readBuffer.ptr<static_cast<host::label_t>(pointerIndex<alpha, coeff>())>()[idxPop<alpha, i, VelocitySet::QF()>(t_a, t_b, b_x, b_y, b_z)]));
                        });
                }
            }

            /**
             * @brief Loads halo population data from neighboring blocks in a specific direction
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam isPeriodic Whether the domain is periodic in this direction
             * @param[out] pop Array to store loaded population values
             * @param[in] readBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             * @param[in] point The global point coordinate
             **/
            template <const axis::type alpha, const bool isPeriodic>
            __device__ static inline constexpr void pull_direction(
                thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, const scalar_t> &readBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                if (boundaryCheck<alpha, -1, isPeriodic>(point.value<alpha>(), Tx))
                {
                    pull_face<alpha, +1>(pop, readBuffer, Tx, Bx);
                }
                else if (boundaryCheck<alpha, +1, isPeriodic>(point.value<alpha>(), Tx))
                {
                    pull_face<alpha, -1>(pop, readBuffer, Tx, Bx);
                }
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks
             * @tparam alpha The axis direction
             * @tparam PtrIndex The index of the pointer corresponding to the halo face
             * @tparam coeff The normal direction; -1 or +1
             * @param[out] pop Array to store loaded population values
             * @param[in] readBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             **/
            template <const axis::type alpha, const int coeff>
            __device__ static inline constexpr void save_face(
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, scalar_t> &writeBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();

                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                device::constexpr_for<0, VelocitySet::QF()>(
                    [&](const auto i)
                    {
                        writeBuffer.ptr<static_cast<host::label_t>(pointerIndex<alpha, coeff>())>()[idxPop<alpha, i, VelocitySet::QF()>(
                            Tx.value<axis::orthogonal<alpha, 0>()>(),
                            Tx.value<axis::orthogonal<alpha, 1>()>(),
                            Bx.value<axis::X>(),
                            Bx.value<axis::Y>(),
                            Bx.value<axis::Z>())] = pop[q_i<streaming_index<alpha, coeff>(i)>()];
                    });
            }

            /**
             * @brief Saves population data to halo regions for neighboring blocks in a specific direction
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam isPeriodic Whether the domain is periodic in this direction
             * @tparam PtrIdx0 The index of the pointer for the negative face halo
             * @tparam PtrIdx1 The index of the pointer for the positive face halo
             * @param[in] pop Array containing population values to save
             * @param[out] writeBuffer Collection of pointers to the halo faces
             * @param[in] Tx Three-dimensional thread coordinates
             * @param[in] Bx Three-dimensional block coordinates
             * @param[in] point The global point coordinate
             **/
            template <const axis::type alpha, const bool isPeriodic>
            __device__ static inline constexpr void save_direction(
                const thread::array<scalar_t, VelocitySet::Q()> &pop,
                const device::ptrCollection<6, scalar_t> &writeBuffer,
                const thread::coordinate &Tx,
                const block::coordinate &Bx,
                const device::pointCoordinate &point) noexcept
            {
                if (boundaryCheck<alpha, -1, isPeriodic>(point.value<alpha>(), Tx))
                {
                    save_face<alpha, -1>(pop, writeBuffer, Tx, Bx);
                }
                else if (boundaryCheck<alpha, +1, isPeriodic>(point.value<alpha>(), Tx))
                {
                    save_face<alpha, +1>(pop, writeBuffer, Tx, Bx);
                }
            }

            /**
             * @brief Returns the pointer index corresponding to the axis direction alpha and coefficient coeff (must be -1 or 1)
             * @tparam alpha The axis direction
             * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
             * @returns The pointer index corresponding to the axis direction alpha and coefficient coeff
             **/
            template <const axis::type alpha, const int coeff>
            __device__ __host__ [[nodiscard]] static inline consteval axis::pointerIndex_t pointerIndex() noexcept
            {
                axis::assertions::validate<alpha, axis::null::NOT_NULL>();
                velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

                if constexpr (alpha == axis::X)
                {
                    if constexpr (coeff == -1)
                    {
                        return axis::pointerIndex_t::West;
                    }
                    if constexpr (coeff == 1)
                    {
                        return axis::pointerIndex_t::East;
                    }
                }

                if constexpr (alpha == axis::Y)
                {
                    if constexpr (coeff == -1)
                    {
                        return axis::pointerIndex_t::South;
                    }
                    if constexpr (coeff == 1)
                    {
                        return axis::pointerIndex_t::North;
                    }
                }

                if constexpr (alpha == axis::Z)
                {
                    if constexpr (coeff == -1)
                    {
                        return axis::pointerIndex_t::Back;
                    }
                    if constexpr (coeff == 1)
                    {
                        return axis::pointerIndex_t::Front;
                    }
                }
            }
        };
    }
}

#endif
