set ( KOTEKAN_OPENCL_SOURCES
        clDeviceInterface.cpp
        clEventContainer.cpp
        clProcess.cpp

# Copy-in & general-purpose:
        clInputData.cpp

# CHIME N2 Kernels & copy-out:
        clOutputDataZero.cpp
        clPresumZero.cpp
        clPresumKernel.cpp
        clPreseedKernel.cpp
        clCorrelatorKernel.cpp
        clOutputData.cpp
        clKVCorr.cpp

# CHIME/Pulsar Kernels & copy-out:
        clBeamformPhaseData.cpp
        clBeamformKernel.cpp
        clOutputBeamformResult.cpp

#RFI Kernels
        #clRfiTimeSum.cpp
        #clRfiInputSum.cpp
        #clRfiOutput.cpp
    )
