/**
 * @file
 * @brief The "gnss" interface document: what the GNSS stages exchange with the broker, the
 *        config generator and the file readers, built from the constants compiled into this
 *        binary.
 *  - build_gnss_interface
 *
 * Registered with REGISTER_KOTEKAN_INTERFACE (interfaceRegistry.hpp), so kotekan serves its
 * digests in /version, the document at GET /interfaces/gnss, and both from
 * `kotekan --interfaces-json`. Its sections:
 *  - signals: every descriptor in ALL_SIGNALS by name, with its band edges at zero Doppler;
 *  - telem: the gnssTelem.hpp constants, the TelemHeader fields and sample frame sizes;
 *  - record.row, record.chan, record.elem, record.cmb: the gnssRecord.hpp slots, each with a
 *    meaning: the unit, then any reference, sign or sentinel, e.g. "rad;nan=no job";
 *  - cube: the beam-cube constants, array order and sample frame sizes;
 *  - gpu_chain: the gnssGpuChain.hpp constants, FrameHdr and PrnCtl fields and sample frame sizes;
 *  - metadata: the GnssChanMetadata wire format;
 *  - enums: Modulation, the REC_PROJ_SRC bits and the PrnCtl::reanchored values;
 *  - rest: a version per family of REST endpoints.
 *
 * lib/stages/gnss/gnss_interface.json is the expected document, and test_gnss_interface checks
 * the build against it. Changing a constant the document holds changes the document: regenerate
 * the golden in the same commit, so its diff shows the interface change.
 *
 *     kotekan --interfaces-json | python3 -c "import json, sys; print(json.dumps(
 *         json.load(sys.stdin)['gnss']['doc'], indent=1, sort_keys=True))" \
 *         > lib/stages/gnss/gnss_interface.json
 *
 * The REST JSON itself is not in the document. When an endpoint adds, removes or renames a key,
 * or changes a unit or meaning, bump its family's version in "rest" (gnssInterface.cpp) by hand.
 */

#ifndef GNSS_INTERFACE_HPP
#define GNSS_INTERFACE_HPP

#include "json.hpp" // for json

namespace gnss {

/// The sections of the "gnss" interface document (schema 1).
nlohmann::json build_gnss_interface();

} // namespace gnss

#endif // GNSS_INTERFACE_HPP
