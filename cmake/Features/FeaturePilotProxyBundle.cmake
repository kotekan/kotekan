# Optional build-time export of the PilotProxy DTV runtime weight bundle (pilot_profiles.json +
# weights.bin), the calibration-data input of the cudaPilotProxyDetector stage (see
# config/fengine/include/dtv_chord.j2).
#
# PILOTPROXY_EXPORT_BUNDLE (declared in Options.cmake):
#
# * OFF (default): do nothing; deployments pin a released bundle instead
# * ON: require the pilot-proxy CLI; export + validate on every build
# * AUTO: export + validate when the CLI is available, skip quietly otherwise
#
# The export is deterministic and fast (~seconds): it synthesizes the integer weight rows for the
# requested ATSC physical-channel range from the packaged receiver / detector-core profiles and
# writes the bundle, then the validator re-derives the norms and checks the manifest. Because it is
# cheap and deterministic, the target re-runs on every build rather than trying to track the state
# of the pip-installed package.
#
# Production note: the runtime bundle is survey calibration data (thresholds, anchors, per-epoch
# fine-calibration status), not a pure build product. Sites should pin a released, validated bundle
# via dtv_runtime_bundle_dir; this target exists so development and test builds are self-contained.

if(NOT "${PILOTPROXY_EXPORT_BUNDLE}" STREQUAL "OFF")
    find_program(PILOTPROXY_CLI pilot-proxy)
    if(NOT PILOTPROXY_CLI)
        if("${PILOTPROXY_EXPORT_BUNDLE}" STREQUAL "ON")
            message(
                FATAL_ERROR
                    "PILOTPROXY_EXPORT_BUNDLE=ON but the pilot-proxy CLI was not found on PATH. "
                    "Install it (pip install from https://github.com/WVURAIL/pilot-proxy) or set "
                    "PILOTPROXY_EXPORT_BUNDLE=AUTO/OFF.")
        else()
            kmsg_status("PilotProxy bundle export skipped (AUTO: pilot-proxy CLI not found).")
        endif()
    else()
        set(PILOTPROXY_BUNDLE_DIR
            "${CMAKE_BINARY_DIR}/pilotproxy_bundle"
            CACHE PATH "Output directory for the exported PilotProxy runtime weight bundle")
        set(PILOTPROXY_CHANNEL_RANGE
            "14:36"
            CACHE STRING "ATSC physical-channel range for the exported PilotProxy bundle")
        # The pip package installs the profile JSONs under <prefix>/share/pilot-proxy/configs/ next
        # to the console script; both cache variables accept explicit paths for out-of-tree
        # profiles.
        get_filename_component(_pilotproxy_bindir "${PILOTPROXY_CLI}" DIRECTORY)
        get_filename_component(_pilotproxy_prefix "${_pilotproxy_bindir}" DIRECTORY)
        set(_pilotproxy_configs "${_pilotproxy_prefix}/share/pilot-proxy/configs")
        set(PILOTPROXY_RECEIVER_PROFILE
            "${_pilotproxy_configs}/receiver_profiles/chord_dtv_fengine.json"
            CACHE FILEPATH "Receiver profile for the exported PilotProxy bundle")
        set(PILOTPROXY_DETECTOR_CORE_PROFILE
            "${_pilotproxy_configs}/detector_core/pilotproxy_cuda_fstat_v1.json"
            CACHE FILEPATH "Detector-core profile for the exported PilotProxy bundle")
        add_custom_target(
            pilotproxy-bundle ALL
            COMMAND
                ${PILOTPROXY_CLI} export-runtime-weight-bundle --receiver-profile
                ${PILOTPROXY_RECEIVER_PROFILE} --detector-core-profile
                ${PILOTPROXY_DETECTOR_CORE_PROFILE} --weight-coordinate-system
                post_spectral_sense_normalization --physical-channel-range
                ${PILOTPROXY_CHANNEL_RANGE} --output-dir ${PILOTPROXY_BUNDLE_DIR}
            COMMAND ${PILOTPROXY_CLI} validate-runtime-weight-bundle --bundle-dir
                    ${PILOTPROXY_BUNDLE_DIR}
            BYPRODUCTS ${PILOTPROXY_BUNDLE_DIR}/pilot_profiles.json
                       ${PILOTPROXY_BUNDLE_DIR}/weights.bin
            COMMENT "Exporting + validating the PilotProxy runtime weight bundle"
            VERBATIM)
        kmsg_status("PilotProxy bundle export enabled -> ${PILOTPROXY_BUNDLE_DIR}")
    endif()
endif()
