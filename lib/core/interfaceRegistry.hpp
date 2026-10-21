/**
 * @file
 * @brief Interface documents: named descriptions of what a kotekan binary
 *        exchanges with programs outside it, built from compiled constants and
 *        served with a digest so a peer can check it speaks the same protocol.
 *  - REGISTER_KOTEKAN_INTERFACE
 *  - register_interface
 *  - build_interfaces
 *  - canonical_json
 *  - canonical_double
 *  - fnv1a64_digest
 *
 * A subsystem registers a builder that returns a JSON object of named
 * sections. kotekan wraps it as the document
 * `{"interface": <name>, "schema": <n>, "sections": {...}}` and serves:
 *  - in `/version` (and `--version-json`), under `"interfaces"`, the schema,
 *    the digest of the document and the digest of each section;
 *  - at `GET /interfaces/<name>`, the document's canonical bytes;
 *  - from `kotekan --interfaces-json`, all of the above without a config.
 *
 * A document holds only objects, arrays, integers that fit in an int64,
 * booleans and strings, with keys and strings in printable ASCII. Floats are
 * refused, since JSON libraries need not write a double the same way: write
 * it with canonical_double() instead. The canonical bytes of a document are what
 * Python's `json.dumps(o, sort_keys=True, separators=(",", ":"),
 * ensure_ascii=True)` gives for it. Changing a document's layout, or this
 * form, means bumping its schema number.
 */

#ifndef INTERFACE_REGISTRY_HPP
#define INTERFACE_REGISTRY_HPP

#include "json.hpp" // for json

#include <stdint.h> // for int64_t
#include <string>   // for string

namespace kotekan {

/**
 * @brief Registers an interface document builder; use REGISTER_KOTEKAN_INTERFACE.
 *
 * A name registered twice is served as an error block.
 *
 * @param name    Interface name, in printable ASCII; also the last part of its
 *                REST endpoint.
 * @param schema  Schema number of the document.
 * @param builder Returns the document's sections as a JSON object.
 * @return 0, so the call can initialise a static.
 */
int register_interface(const std::string& name, int64_t schema, nlohmann::json (*builder)());

/**
 * @brief Runs every registered builder.
 *
 * Never throws. An interface whose builder throws, returns something other
 * than an object, or returns a value outside the canonical form is given as
 * `{"error": "<message>"}`.
 *
 * @return `{<name>: {"schema": n, "digest": d, "sections": {<section>: d},
 *         "doc": <document>}}`, an empty object if nothing is registered. Each
 *         digest is fnv1a64_digest() of the canonical bytes of the document
 *         or of the section.
 */
nlohmann::json build_interfaces();

/**
 * @brief The canonical bytes of a JSON value.
 *
 * @param value A value in the canonical form described in this file.
 * @return The compact dump with sorted keys.
 * @throws std::invalid_argument naming the path of the first value outside
 *         the canonical form.
 */
std::string canonical_json(const nlohmann::json& value);

/**
 * @brief A double as the string `printf("%.17g")` writes, which reads back as
 *        the same double. Python's `"%.17g" % x` gives the same string.
 */
std::string canonical_double(double value);

/**
 * @brief The FNV-1a 64-bit hash of a byte string, as `fnv1a64:` and 16
 *        lowercase hex digits.
 *
 * FNV-1a rather than MurmurHash3 (Hash.hpp) or MD5: a peer computes it in a
 * few lines of plain Python, and OpenSSL is optional in kotekan_core.
 */
std::string fnv1a64_digest(const std::string& bytes);

} // namespace kotekan

/**
 * @brief Registers an interface document builder at static initialisation.
 *
 * Use in the .cpp file that defines the builder, e.g.
 * `REGISTER_KOTEKAN_INTERFACE("gnss", 1, build_gnss_interface);`
 *
 * A builder reports a problem by throwing, which gives its interface an error
 * block. It must not call FATAL_ERROR: that raises SIGTERM before it throws,
 * so kotekan stops even though the registry catches the throw.
 *
 * @param name    Interface name, a string.
 * @param schema  Schema number of the document.
 * @param builder Unqualified name of a function `nlohmann::json f()` that
 *                returns the document's sections.
 */
#define REGISTER_KOTEKAN_INTERFACE(name, schema, builder)                                          \
    __attribute__((used)) static int _kotekan_interface_##builder =                                \
        ::kotekan::register_interface(name, schema, builder)

#endif /* INTERFACE_REGISTRY_HPP */
