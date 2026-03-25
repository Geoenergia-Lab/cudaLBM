/**
Filename: fieldAverage.cuh
Contents: Implements time averaging of the solution variables
**/

#ifndef __MBLBM_FIELDAVERAGE_CUH
#define __MBLBM_FIELDAVERAGE_CUH

#include "LBMIncludes.cuh"
#include "typedefs/typedefs.cuh"

namespace LBM
{
    namespace fieldAverage
    {
        /**
         * @brief Executes the time average of all variables within fMom
         * @param[in] fMom The 10 solution variables
         * @param[in] fMomMean The time averaged solution variables
         * @param[in] nodeTypes The node types of the mesh
         **/
        launchBounds __global__ void calculate(
            const scalar_t *const ptrRestrict fMom,
            scalar_t *const ptrRestrict fMomMean,
            const device::label_t step)
        {
            if (device::out_of_bounds())
            {
                return;
            }

            const device::label_t meanCounter = static_cast<scalar_t>(step);

            const scalar_t invCount = static_cast<scalar_t>(1.0) / (meanCounter + static_cast<scalar_t>(1.0));

            // Load the instantaneous moments
            const thread::array<scalar_t, NUMBER_MOMENTS()> moments = {
                fMom[device::idxMom<index::rho>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::u>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::v>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::w>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::xx>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::xy>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::xz>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::yy>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::yz>(threadIdx, blockIdx)],
                fMom[device::idxMom<index::zz>(threadIdx, blockIdx)]};

            // Load the mean moments
            const thread::array<scalar_t, NUMBER_MOMENTS()> meanMoments = {
                fMomMean[device::idxMom<index::rho>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::u>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::v>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::w>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::xx>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::xy>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::xz>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::yy>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::yz>(threadIdx, blockIdx)],
                fMomMean[device::idxMom<index::zz>(threadIdx, blockIdx)]};

            // Write into memory
            fMomMean[device::idxMom<index::rho>(threadIdx, blockIdx)] = ((meanMoments.arr[index::rho] * meanCounter) + moments.arr[index::rho]) * invCount;
            fMomMean[device::idxMom<index::u>(threadIdx, blockIdx)] = ((meanMoments.arr[index::u] * meanCounter) + moments.arr[index::u]) * invCount;
            fMomMean[device::idxMom<index::v>(threadIdx, blockIdx)] = ((meanMoments.arr[index::v] * meanCounter) + moments.arr[index::v]) * invCount;
            fMomMean[device::idxMom<index::w>(threadIdx, blockIdx)] = ((meanMoments.arr[index::w] * meanCounter) + moments.arr[index::w]) * invCount;
            fMomMean[device::idxMom<index::xx>(threadIdx, blockIdx)] = ((meanMoments.arr[index::xx] * meanCounter) + moments.arr[index::xx]) * invCount;
            fMomMean[device::idxMom<index::xy>(threadIdx, blockIdx)] = ((meanMoments.arr[index::xy] * meanCounter) + moments.arr[index::xy]) * invCount;
            fMomMean[device::idxMom<index::xz>(threadIdx, blockIdx)] = ((meanMoments.arr[index::xz] * meanCounter) + moments.arr[index::xz]) * invCount;
            fMomMean[device::idxMom<index::yy>(threadIdx, blockIdx)] = ((meanMoments.arr[index::yy] * meanCounter) + moments.arr[index::yy]) * invCount;
            fMomMean[device::idxMom<index::yz>(threadIdx, blockIdx)] = ((meanMoments.arr[index::yz] * meanCounter) + moments.arr[index::yz]) * invCount;
            fMomMean[device::idxMom<index::zz>(threadIdx, blockIdx)] = ((meanMoments.arr[index::zz] * meanCounter) + moments.arr[index::zz]) * invCount;
        }
    }
}

#endif