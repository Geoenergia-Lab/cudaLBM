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
    File containing a list of all valid function object names

Namespace
    LBM::host

SourceFiles
    objectRegistry.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_OBJECTREGISTRY_CUH
#define __MBLBM_OBJECTREGISTRY_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "../strings.cuh"
#include "../postProcess/postProcess.cuh"
#include "functionObjects.cuh"
#include "functionObjectBase.cuh"
#include "moments.cuh"
#include "strainRateTensor.cuh"
#include "kineticEnergy.cuh"
#include "rhoMean.cuh"
#include "UMean.cuh"
#include "PiMean.cuh"

namespace LBM
{
    /**
     * @brief Registry for managing function objects and their calculations
     * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
     **/
    template <class VelocitySet>
    class objectRegistry
    {
    public:
        /**
         * @brief Constructs an objectRegistry with mesh, device pointers and streams
         * @param[in] mesh The lattice mesh
         * @param[in] devPtrs Device pointer collection for memory management
         * @param[in] streamsLBM Stream handler for LBM operations
         **/
        __host__ [[nodiscard]] objectRegistry(
            host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
            const host::latticeMesh &mesh,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
            const streamHandler &streamsLBM,
            const programControl &programCtrl)
            : hostWriteBuffer_(hostWriteBuffer),
              mesh_(mesh),
              rho_(hostWriteBuffer, mesh, rho, U, Pi, streamsLBM, programCtrl),
              U_(hostWriteBuffer, mesh, rho, U, Pi, streamsLBM, programCtrl),
              Pi_(hostWriteBuffer, mesh, rho, U, Pi, streamsLBM, programCtrl),
              S_(hostWriteBuffer, mesh, rho, U, Pi, streamsLBM, programCtrl),
              k_(hostWriteBuffer, mesh, rho, U, Pi, streamsLBM, programCtrl),
              functionVector_(functionObjectCallInitialiser(rho_, U_, Pi_, S_, k_)),
              saveVector_(functionObjectSaveInitialiser(rho_, U_, Pi_, S_, k_)) {}

        /**
         * @brief Default destructor
         **/
        ~objectRegistry() {}

        /**
         * @brief Disable copying
         **/
        __host__ [[nodiscard]] objectRegistry(const objectRegistry &) = delete;
        __host__ [[nodiscard]] objectRegistry &operator=(const objectRegistry &) = delete;

        /**
         * @brief Executes all registered function object calculations for given time step
         * @param[in] timeStep The current simulation time step
         **/
        inline void calculate() noexcept
        {
            for (const auto &func : functionVector_)
            {
                func(); // Call each function with the timeStep
            }
        }

        /**
         * @brief Executes all registered function object calculations for given time step
         * @param[in] timeStep The current simulation time step
         **/
        inline void save(const host::label_t timeStep) noexcept
        {
            for (const auto &save : saveVector_)
            {
                save(timeStep); // Call each function with the timeStep
            }
        }

    private:
        host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer_;

        /**
         * @brief Reference to lattice mesh
         **/
        const host::latticeMesh &mesh_;

        /**
         * @brief Moments
         **/
        functionObjects::density<VelocitySet> rho_;
        functionObjects::velocityVector<VelocitySet> U_;
        functionObjects::secondOrderMoments<VelocitySet> Pi_;

        /**
         * @brief Strain rate tensor function object
         **/
        functionObjects::strainRateTensor<VelocitySet> S_;

        /**
         * @brief Kinetic energy function object
         **/
        functionObjects::kineticEnergy<VelocitySet> k_;

        /**
         * @brief Registry of function objects to invoke
         **/
        const std::vector<functionObjects::calculateFunction> functionVector_;

        /**
         * @brief Initializes function calls based on strain rate tensor configuration
         * @param[in] args References to the objects contained in the registry
         * @return Vector of function objects to be executed
         **/
        template <typename... Args>
        __host__ [[nodiscard]] static const std::vector<functionObjects::calculateFunction> functionObjectCallInitialiser(Args &...args) noexcept
        {
            std::vector<functionObjects::calculateFunction> calls;
            (addObjectCall(calls, args), ...);
            return calls;
        }

        template <class C>
        __host__ static void addObjectCall(std::vector<functionObjects::calculateFunction> &calls, C &object) noexcept
        {
            // If both instantaneous and mean calculations are enabled, calculate both in one call
            // Only do this for variables other than the 10 moments
            if constexpr (C::ObjectType::canCalculateInstantaneous)
            {
                if ((object.doInstantaneous()) && (object.doMean()))
                {
                    calls.push_back(
                        [&object]()
                        {
                            object.calculateInstantaneousAndMean();
                        });
                }
            }

            // Must be only saving instantaneous, so just calculate instantaneous without saving mean
            if constexpr (C::ObjectType::canCalculateInstantaneous)
            {
                if (object.doInstantaneous() && !(object.doMean()))
                {
                    calls.push_back(
                        [&object]()
                        {
                            object.calculateInstantaneous();
                        });
                }
            }

            // Must be only saving the mean, so just calculate mean without saving instantaneous
            if (object.doMean() && !(object.doInstantaneous()))
            {
                calls.push_back(
                    [&object]()
                    {
                        object.calculateMean();
                    });
            }
        }

        /**
         * @brief Registry of function objects to save
         **/
        const std::vector<functionObjects::saveFunction> saveVector_;

        /**
         * @brief Initializes save calls based on strain rate tensor configuration
         * @param[in] args References to the objects contained in the registry
         * @return Vector of function objects to be executed
         **/
        template <typename... Args>
        __host__ [[nodiscard]] static const std::vector<functionObjects::saveFunction> functionObjectSaveInitialiser(Args &...args) noexcept
        {
            std::vector<functionObjects::saveFunction> calls;
            (addSaveCall(calls, args), ...);
            return calls;
        }

        template <class C>
        __host__ static void addSaveCall(std::vector<functionObjects::saveFunction> &calls, C &object) noexcept
        {
            if constexpr (C::ObjectType::canCalculateInstantaneous)
            {
                if (object.doInstantaneous())
                {
                    calls.push_back(
                        [&object](const host::label_t label)
                        {
                            object.saveInstantaneous(label);
                        });
                }
            }
            if (object.doMean())
            {
                calls.push_back(
                    [&object](const host::label_t label)
                    {
                        object.saveMean(label);
                    });
            }
        }
    };
}

#endif