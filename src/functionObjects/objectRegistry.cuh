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
#include "../LBMTypedefs.cuh"
#include "../strings.cuh"
#include "functionObjects.cuh"
#include "moments.cuh"
#include "strainRateTensor.cuh"
#include "kineticEnergy.cuh"

namespace LBM
{
    /**
     * @brief Registry for managing function objects and their calculations
     * @tparam VelocitySet The velocity set type used in LBM simulations
     * @tparam N The number of streams (as a compile-time constant)
     **/
    template <class VelocitySet, const label_t N>
    class objectRegistry
    {
    public:
        /**
         * @brief Constructs an objectRegistry with mesh, device pointers and streams
         * @param[in] mesh Reference to the lattice mesh
         * @param[in] devPtrs Device pointer collection for memory management
         * @param[in] streamsLBM Stream handler for LBM operations
         **/
        [[nodiscard]] objectRegistry(
            host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
            const host::latticeMesh &mesh,
            const device::ptrCollection<10, scalar_t> &devPtrs,
            const streamHandler &streamsLBM)
            : hostWriteBuffer_(hostWriteBuffer),
              mesh_(mesh),
              M_(hostWriteBuffer, mesh, devPtrs, streamsLBM),
              S_(hostWriteBuffer, mesh, devPtrs, streamsLBM),
              k_(hostWriteBuffer, mesh, devPtrs, streamsLBM),
              functionVector_(functionObjectCallInitialiser(M_, S_, k_)),
              saveVector_(functionObjectSaveInitialiser(M_, S_, k_)) {};

        /**
         * @brief Default destructor
         **/
        ~objectRegistry() {};

        /**
         * @brief Executes all registered function object calculations for given time step
         * @param[in] timeStep The current simulation time step
         **/
        inline void calculate(const label_t timeStep) noexcept
        {
            // std::cout << "Length of functionVector_: " << functionVector_.size() << std::endl;
            for (const auto &func : functionVector_)
            {
                func(timeStep); // Call each function with the timeStep
            }
        }

        /**
         * @brief Executes all registered function object calculations for given time step
         * @param[in] timeStep The current simulation time step
         **/
        inline void save(const label_t timeStep) noexcept
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
         * @brief Moments function object
         **/
        functionObjects::moments::collection<VelocitySet> M_;

        /**
         * @brief Strain rate tensor function object
         **/
        functionObjects::strainRate::tensor<VelocitySet> S_;

        /**
         * @brief Kinetic energy function object
         **/
        functionObjects::kineticEnergy::scalar<VelocitySet> k_;

        /**
         * @brief Registry of function objects to invoke
         **/
        const std::vector<std::function<void(const label_t)>> functionVector_;

        /**
         * @brief Initializes function calls based on strain rate tensor configuration
         * @param[in] S Reference to strain rate tensor object
         * @return Vector of function objects to be executed
         **/
        __host__ [[nodiscard]] const std::vector<std::function<void(const label_t)>> functionObjectCallInitialiser(
            functionObjects::moments::collection<VelocitySet> &moments,
            functionObjects::strainRate::tensor<VelocitySet> &S,
            functionObjects::kineticEnergy::scalar<VelocitySet> &k) const noexcept
        {
            std::vector<std::function<void(const label_t)>> calls;

            addObjectCall(calls, moments);
            addObjectCall(calls, S);
            addObjectCall(calls, k);

            return calls;
        }

        template <class C>
        __host__ void addObjectCall(std::vector<std::function<void(const label_t)>> &calls, C &object) const noexcept
        {
            // If both instantaneous and mean calculations are enabled, calculate both in one call
            // Only do this for variables other than the 10 moments
            if constexpr (!std::is_same_v<C, functionObjects::moments::collection<VelocitySet>>)
            {
                if ((object.calculate()) && (object.calculateMean()))
                {
                    calls.push_back(
                        [&object](const label_t label)
                        { object.calculateInstantaneousAndMean(label); });
                }
            }

            // Must be only saving instantaneous, so just calculate instantaneous without saving mean
            if constexpr (!std::is_same_v<C, functionObjects::moments::collection<VelocitySet>>)
            {
                if (object.calculate() && !(object.calculateMean()))
                {
                    calls.push_back(
                        [&object](const label_t label)
                        { object.calculateInstantaneous(label); });
                }
            }

            // Must be only saving the mean, so just calculate mean without saving instantaneous
            if (object.calculateMean() && !(object.calculate()))
            {
                // std::cout << "Pushing back " << object.fieldName() << ".saveMean" << std::endl;
                calls.push_back(
                    [&object](const label_t label)
                    { object.calculateMean(label); });
            }
        }

        /**
         * @brief Registry of function objects to save
         **/
        const std::vector<std::function<void(const label_t)>> saveVector_;

        /**
         * @brief Initializes save calls based on strain rate tensor configuration
         * @param[in] S Reference to strain rate tensor object
         * @return Vector of function objects to be executed
         **/
        __host__ [[nodiscard]] const std::vector<std::function<void(const label_t)>> functionObjectSaveInitialiser(
            functionObjects::moments::collection<VelocitySet> &moments,
            functionObjects::strainRate::tensor<VelocitySet> &S,
            functionObjects::kineticEnergy::scalar<VelocitySet> &k) const noexcept
        {
            std::vector<std::function<void(const label_t)>> calls;

            addSaveCall(calls, moments);
            addSaveCall(calls, S);
            addSaveCall(calls, k);

            return calls;
        }

        template <class C>
        __host__ void addSaveCall(std::vector<std::function<void(const label_t)>> &calls, C &object) const noexcept
        {
            if constexpr (!std::is_same_v<C, functionObjects::moments::collection<VelocitySet>>)
            {
                if (object.calculate())
                {
                    // std::cout << "Pushing back saveInstantaneous" << std::endl;
                    calls.push_back(
                        [&object](const label_t label)
                        { object.saveInstantaneous(label); });
                }
            }
            if (object.calculateMean())
            {
                // std::cout << "Pushing back " << object.fieldName() << ".calculateMean" << std::endl;
                calls.push_back(
                    [&object](const label_t label)
                    { object.saveMean(label); });
            }
        }
    };
}

#endif