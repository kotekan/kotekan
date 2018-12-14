set (KOTEKAN_HSA_SOURCES
        hsaEventContainer.cpp
        hsaBase.c
        hsaCommand.cpp
        hsaDeviceInterface.cpp
        hsaProcess.cpp
        hsaSubframeCommand.cpp

# Copy-in & general-purpose:
        hsaInputData.cpp
        hsaInputLostSamples.cpp
        hsaBarrier.cpp

# RFI Kernels & copy-out:
        hsaRfiVdif.cpp
        hsaRfiTimeSum.cpp
        hsaRfiInputSum.cpp
        hsaRfiBadInput.cpp
        hsaRfiZeroData.cpp
        hsaRfiBadInputOutput.cpp
        hsaRfiMaskOutput.cpp
        hsaRfiOutput.cpp

# CHIME N2 Kernels & copy-out:
        hsaOutputDataZero.cpp
        hsaPresumZero.cpp
        hsaPresumKernel.cpp
        hsaCorrelatorKernel.cpp
        hsaOutputData.cpp

# CHIME/FRB Kernels & copy-out:
        hsaBeamformReorder.cpp
        hsaBeamformKernel.cpp
        hsaBeamformTranspose.cpp
        hsaBeamformUpchan.cpp
        hsaBeamformOutput.cpp

# CHIME/Pulsar Kernels & copy-out:
        hsaBeamformPulsar.cpp
        hsaBeamformPulsarOutput.cpp
        hsaPulsarUpdatePhase.cpp
)
