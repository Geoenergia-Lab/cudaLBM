/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
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
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
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
    Definition of the main GPU kernel

Namespace
    LBM

SourceFiles
    deviceCommunicator.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICECOMMUNICATOR_CUH
#define __MBLBM_DEVICECOMMUNICATOR_CUH

namespace LBM
{
    template <class VelocitySet>
    class deviceCommunicator
    {
        using This = deviceCommunicator<VelocitySet>;
        using exchangeFunction = std::function<void(const host::label_t, const host::label_t)>;

    public:
        /**
         * @brief Construct a deviceCommunicator object from the mesh, program control and halo pointers
         * @param[in] mesh The lattice mesh
         * @param[in] programCtrl The program control object
         * @param[in] haloPtrs The halo to exchange between devices
         **/
        __host__ [[nodiscard]] deviceCommunicator(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const haloBuffer<VelocitySet> &haloPtrs) noexcept
            : mesh_(mesh),
              programCtrl_(programCtrl),
              haloPtrs_(haloPtrs),
              commList_(assembleCommList(programCtrl)) {}

        /**
         * @brief Destructor
         **/
        __host__ ~deviceCommunicator() noexcept {}

        /**
         * @brief Disable copying
         **/
        __host__ [[nodiscard]] deviceCommunicator(const deviceCommunicator &) = delete;
        __host__ [[nodiscard]] deviceCommunicator &operator=(const deviceCommunicator &) = delete;

        /**
         * @brief Perform the inter-device exchange for the given time step
         * @param[in] timeStep The current time step
         **/
        __host__ inline void exchange(const host::label_t timeStep) const noexcept
        {
            for (host::label_t idxPair = 0; idxPair < commList_.size(); idxPair++)
            {
                commList_[idxPair](idxPair, timeStep);
            }
        }

    private:
        /**
         * @brief Reference to the lattice mesh
         **/
        const host::latticeMesh &mesh_;

        /**
         * @brief Reference to program control
         **/
        const programControl &programCtrl_;

        /**
         * @brief Reference to the device halo
         **/
        const haloBuffer<VelocitySet> &haloPtrs_;

        /**
         * @brief List of exchange functions to execute per time step
         **/
        const std::vector<exchangeFunction> commList_;

        /**
         * @brief Assemble the list of exchange functions from the program control object
         * @param[in] programCtrl The program control object
         * @return A std::vector of exchange functions to be called at run time
         **/
        __host__ [[nodiscard]] const std::vector<exchangeFunction> assembleCommList(const programControl &programCtrl) const noexcept
        {
            std::vector<exchangeFunction> commList;

            if (programCtrl.deviceList().size() > 1)
            {
                for (host::label_t idxPair = 0; idxPair < programCtrl.deviceList().size() - 1; idxPair++)
                {
                    commList.push_back(
                        [this](const host::label_t pair, const host::label_t timeStep)
                        {
                            this->exchangeImpl<axis::Z>(pair, timeStep);
                        });
                }
            }

            return commList;
        }

        /**
         * @brief Get the relevant starting block index for a particular axis
         * @tparam alpha The axis direction (X, Y or Z)
         * @param[in] mesh The lattice mesh
         **/
        template <const axis::type alpha>
        __host__ [[nodiscard]] static inline constexpr host::blockLabel commBlockID(const host::latticeMesh &mesh) noexcept
        {
            if constexpr (alpha == axis::X)
            {
                return host::blockLabel(mesh.blocksPerDevice<alpha>() - 1, 0, 0);
            }

            if constexpr (alpha == axis::Y)
            {
                return host::blockLabel(0, mesh.blocksPerDevice<alpha>() - 1, 0);
            }

            if constexpr (alpha == axis::Z)
            {
                return host::blockLabel(0, 0, mesh.blocksPerDevice<alpha>() - 1);
            }
        }

        /**
         * @brief Implementation of the exchange function
         * @tparam alpha The axis direction (X, Y or Z)
         * @param[in] idxExchange The ID of the bidirectional exchange
         * @param[in] timeStep The current time step
         **/
        template <const axis::type alpha>
        __host__ inline void exchangeImpl(const host::label_t idxExchange, const host::label_t timeStep) const noexcept
        {
            static_assert(alpha == axis::Z, "HermiteLBM currently only supports decomposition in the z axis");

            const host::label_t nab = mesh_.nBlocks<axis::orthogonal<alpha, 0>()>();
            const host::label_t nbb = mesh_.nBlocks<axis::orthogonal<alpha, 1>()>();

            constexpr const host::threadLabel threadStart(static_cast<device::label_t>(0), static_cast<device::label_t>(0), static_cast<device::label_t>(0));

            const host::label_t idxDevL = idxExchange;
            const host::label_t idxDevR = idxExchange + 1;

            // Right to Left exchange
            constexpr const host::blockLabel blockIdxDestL(0, 0, 0);
            const host::label_t idxDestL = host::idxPop<alpha, VelocitySet::template QF<host::label_t>()>(0, threadStart, blockIdxDestL, nab, nbb);
            constexpr const host::blockLabel RDeviceSourceBlock(0, 0, 0);
            const host::label_t idxSrcR = host::idxPop<alpha, VelocitySet::template QF<host::label_t>()>(0, threadStart, RDeviceSourceBlock, nab, nbb);

            // Left to Right exchange
            const host::blockLabel blockIdxDestR = This::commBlockID<alpha>(mesh_);
            const host::label_t idxDestR = host::idxPop<alpha, VelocitySet::template QF<host::label_t>()>(0, threadStart, blockIdxDestR, nab, nbb);
            const host::blockLabel LDeviceSourceBlock = This::commBlockID<alpha>(mesh_);
            const host::label_t idxSrcL = host::idxPop<alpha, VelocitySet::template QF<host::label_t>()>(0, threadStart, LDeviceSourceBlock, nab, nbb);

            // Call the exchange functions
            const host::label_t area = VelocitySet::template QF<host::label_t>() * block::n<axis::orthogonal<alpha, 0>(), host::label_t>() * block::n<axis::orthogonal<alpha, 1>(), host::label_t>() * mesh_.blocksPerDevice<axis::orthogonal<alpha, 0>()>() * mesh_.blocksPerDevice<axis::orthogonal<alpha, 1>()>();

            This::exchange<alpha, -1>(idxDevR, idxDevL, idxSrcR, idxDestL, haloPtrs_, programCtrl_, area, timeStep); // Copy to the Left GPU
            This::exchange<alpha, +1>(idxDevL, idxDevR, idxSrcL, idxDestR, haloPtrs_, programCtrl_, area, timeStep); // Copy to the Right GPU
        }

        /**
         * @brief Helper function for peer-to-peer memory exchange
         * @tparam alpha The axis direction (X, Y or Z)
         * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
         * @param[in] timeStep The current time step
         **/
        template <const axis::type alpha, const int coeff>
        __host__ static inline void exchange(
            const host::label_t idxDevSrc,
            const host::label_t idxDevDst,
            const host::label_t idxSrc,
            const host::label_t idxDst,
            const haloBuffer<VelocitySet> &haloPtrs,
            const programControl &programCtrl,
            const host::label_t nPoints,
            const host::label_t timeStep) noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            device::memcpyPeerAsync(
                &(haloPtrs.writeBuffer(idxDevDst, timeStep).template ptr<device::pointerIndex<alpha, coeff>()>()[idxDst]),
                programCtrl.deviceList()[idxDevDst],
                &(haloPtrs.writeBuffer(idxDevSrc, timeStep).template ptr<device::pointerIndex<alpha, coeff>()>()[idxSrc]),
                programCtrl.deviceList()[idxDevSrc],
                nPoints,
                programCtrl.streams()[device::idxStream<coeff>(idxDevDst)]);
        }
    };
}

#endif