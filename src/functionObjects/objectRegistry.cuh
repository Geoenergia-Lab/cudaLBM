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
    File containing a list of all valid function object names

Namespace
    LBM::host

SourceFiles
    objectRegistry.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_OBJECTREGISTRY_CUH
#define __MBLBM_OBJECTREGISTRY_CUH

#include "../postProcess/postProcess.cuh"
#include "../momentBasedLBM/ptrCollection.cuh"
#include "functionObjects.cuh"
#include "functionObjectBase.cuh"
#include "objects/objects.cuh"
#include "turbulenceStatistics.cuh"

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
         * @param[in] rho Device scalar field containing the density values on the GPU
         * @param[in] U Device vector field containing the velocity values on the GPU
         * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
         * @param[in] programCtrl The program control object
         **/
        __host__ [[nodiscard]] objectRegistry(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const kernel::ptrCollection &devPtrs)
            : mesh_(mesh),
              rho_(mesh, devPtrs, programCtrl),
              U_(mesh, devPtrs, programCtrl),
              Pi_(mesh, devPtrs, programCtrl),
              S_(mesh, devPtrs, programCtrl),
              k_(mesh, devPtrs, programCtrl),
              functionVector_(functionObjectCallInitialiser(rho_, U_, Pi_, S_, k_)),
              saveVector_(functionObjectSaveInitialiser(rho_, U_, Pi_, S_, k_)) {}

        /**
         * @brief Default destructor
         **/
        __host__ ~objectRegistry() {}

        /**
         * @brief Disable copying
         **/
        __host__ [[nodiscard]] objectRegistry(const objectRegistry &) = delete;
        __host__ [[nodiscard]] objectRegistry &operator=(const objectRegistry &) = delete;

        /**
         * @brief Executes all registered function object calculations for given time step
         * @param[in] timeStep The current simulation time step
         **/
        __host__ inline void calculate() noexcept
        {
            for (const functionObjects::calculateFunction &func : functionVector_)
            {
                func(); // Call each function with the timeStep
            }
        }

        /**
         * @brief Executes all registered function object calculations for given time step
         * @param[in] timeStep The current simulation time step
         **/
        __host__ inline void save(host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep) noexcept
        {
            for (const functionObjects::saveFunction &save : saveVector_)
            {
                save(hostWriteBuffer, timeStep); // Call each function with the timeStep
            }
        }

    private:
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

        /**
         * @brief Adds a call to a calculate function to the list of functions to call
         * @param[out] calls The list of functions to be called
         * @param[in] object The object whose calculate function to add
         **/
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

            // Push back the call to calculate the mean quantity
            if (object.doPrime())
            {
                calls.push_back(
                    [&object]()
                    {
                        object.calculatePrime();
                    });
            }

            if (object.doPrimeSqMean())
            {
                calls.push_back(
                    [&object]()
                    {
                        object.calculatePrimeSqMean();
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

        /**
         * @brief Adds a call to a save function to the list of functions to call
         * @param[out] calls The list of functions to be called
         * @param[in] object The object whose save function to add
         **/
        template <class C>
        __host__ static void addSaveCall(std::vector<functionObjects::saveFunction> &calls, C &object) noexcept
        {
            if constexpr (C::ObjectType::canCalculateInstantaneous)
            {
                if (object.doInstantaneous())
                {
                    calls.push_back(
                        [&object](host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep)
                        {
                            object.saveInstantaneous(hostWriteBuffer, timeStep);
                        });
                }
            }
            if (object.doMean() || object.doPrime() || object.doPrimeSqMean())
            {
                calls.push_back(
                    [&object](host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep)
                    {
                        object.saveMean(hostWriteBuffer, timeStep);
                    });
            }
            if (object.doPrime())
            {
                calls.push_back(
                    [&object](host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep)
                    {
                        object.savePrime(hostWriteBuffer, timeStep);
                    });
            }

            if (object.doPrimeSqMean())
            {
                calls.push_back(
                    [&object](host::array<host::PINNED, scalar_t> &hostWriteBuffer, const host::label_t timeStep)
                    {
                        object.savePrimeSqMean(hostWriteBuffer, timeStep);
                    });
            }
        }
    };
}

#endif