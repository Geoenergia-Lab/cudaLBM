/**
Filename: mpiStatus.cuh
Contents: A class handling the initialisation of CUDA-aware MPI
**/

#ifndef __MBLBM_MPISTATUS_CUH
#define __MBLBM_MPISTATUS_CUH

#include "LBMIncludes.cuh"
#include "typedefs/typedefs.cuh"

namespace LBM
{
    class mpiStatus
    {
    public:
        /**
         * @brief Constructor for the mpiStatus class
         **/
        __host__ [[nodiscard]] mpiStatus(int argc, char *argv[])
            : mpiStatus_(mpiInitialise(argc, argv))
        {
            if (!mpiStatus_ == MPI_SUCCESS)
            {
                throw std::runtime_error("MPI failed to initialise, returned code " + std::to_string(static_cast<int>(mpiStatus_)));
            }
        };

        /**
         * @brief Destructor for the mpiStatus class
         **/
        ~mpiStatus()
        {
            const mpiError_t i = static_cast<mpiError_t>(MPI_Finalize());

            if (!i == mpiError_t::SUCCESS)
            {
                throw std::runtime_error("MPI failed to finalise, returned code " + std::to_string(static_cast<int>(i)));
            }
        };

        /**
         * @brief Returns the MPI status
         * @return The MPI status
         **/
        __host__ [[nodiscard]] inline mpiError_t status() const noexcept
        {
            return mpiStatus_;
        }

    private:
        /**
         * @brief The MPI status
         **/
        const mpiError_t mpiStatus_;

        /**
         * @brief Initialises MPI from the input arguments
         * @return The MPI status
         **/
        __host__ [[nodiscard]] mpiError_t mpiInitialise(int argc, char **argv) const noexcept
        {
            return static_cast<mpiError_t>(MPI_Init(&argc, &argv));
        }
    };
}

#endif