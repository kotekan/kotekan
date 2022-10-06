// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Kotekan Developers

/****************************************************
 * @file   metadatFactory.hpp
 * @brief  This file declares the factory for making
 *         metadata structures, registered previously.
 *         Also, contains MetadataFactory and associated
 *         helper classes and templates:
 *          - metadataMaker
 *          - metadataFactory
 *          - metadataFactoryRegistry
 *          - metadataMakerTemplate<T>
 *
 * @author Mehdi Najafi
 * @date   28 AUG 2022
 *****************************************************/

#ifndef METADATA_FACTORY_HPP
#define METADATA_FACTORY_HPP

#include "Config.hpp"         // for Config
#include "kotekanLogging.hpp" // for ERROR_NON_OO, INFO_NON_OO
#include "metadata.h"         // for create_metadata_pool, metadataPool

#include "json.hpp" // for json

#include <exception> // for exception
#include <map>       // for map, _Rb_tree_iterator
#include <regex>     // for match_results<>::_Base_type
#include <stddef.h>  // for size_t
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint32_t
#include <string>    // for string
#include <vector>    // for vector

namespace kotekan {

/**
 * @struct metadataMaker
 * @brief  A pure abstract metadata maker structure,
 *         used as a basis for every metadata structure.
 *
 * @author Mehdi Najafi
 */
struct metadataMaker {
public:
    virtual void* create() const = 0;
    virtual size_t size() const = 0;
};

/**
 * @struct  metadataRegistry
 * @brief   A registration structure for all metadata structures.
 *
 * @author Mehdi Najafi
 */
struct metadataRegistry {
    /// the string to metadataMaker map, the only member variable
    std::map<std::string, metadataMaker*> registered_metadatas;

    /// check if a given name is a registered metadata
    bool isMetadataRegistered(const std::string& key) {
        return registered_metadatas.find(key) != registered_metadatas.end();
    }

    /// add a metadataMaker with a given string key to the map
    void record(const std::string& key, metadataMaker* proc) {
        if (isMetadataRegistered(key)) {
            // do nothing and no error since the metadata header file
            // can be included in multiple source files
            return;
        }
        registered_metadatas[key] = proc;
    }

    /// retrieve a metadataMaker associated with a given string key in the map
    metadataMaker* retrieve(const std::string& key) {
        auto proc = registered_metadatas.find(key);
        if (proc == registered_metadatas.end()) {
            ERROR_NON_OO("Unrecognized metadata! ({:s})", key);
            throw std::runtime_error(
                "Tried to instantiate a metadata object which is not registered!");
        }
        return registered_metadatas[key];
    }

    /// retrieve the single instance of the metadataRegistry
    static metadataRegistry& instance() {
        static metadataRegistry registerer;
        return registerer;
    }

    /// add a metadataMaker with a given string key to the map
    /// on the single instance of the metadataRegistry
    static void kotekan_reg_record(const std::string& key, metadataMaker* proc) {
        instance().record(key, proc);
    }

    /// retrieve a metadataMaker associated with a given string key in the map
    /// on the single instance of the metadataRegistry
    static metadataMaker* kotekan_reg_retrieve(const std::string& key) {
        return instance().retrieve(key);
    }

    /// check if the metadata is already registered or not
    static bool kotekan_isMetadataRegistered(const std::string& key) {
        return instance().isMetadataRegistered(key);
    }
};

/**
 * @struct  metadataMakerTemplate
 * @brief   A metadata maker structure used for all metadata structures.
 *
 * @author  Mehdi Najafi
 */
template<typename T>
struct metadataMakerTemplate : metadataMaker {
    metadataMakerTemplate(const std::string& key) {
        metadataRegistry::kotekan_reg_record(key, this);
    }
    virtual void* create() const override {
        return new T();
    }
    virtual size_t size() const override {
        return sizeof(T);
    }
};

#ifndef REGISTER_KOTEKAN_METADATA
/// special macro for registration of metadata structures
#define REGISTER_KOTEKAN_METADATA(T) const ::kotekan::metadataMakerTemplate<T> local_maker_##T(#T);
#endif // REGISTER_KOTEKAN_METADATA


/**
 * @class metadataFactory
 * @brief A metadata factory to obtain metadata structure based
 *        on given name as a string.
 *
 * @author Mehdi Najafi
 */

class metadataFactory {

public:
    /// One metadataFactory should be created for a config
    metadataFactory(Config& config) : config(config) {}

    ~metadataFactory() {}

    /// Creates all the metadatas listed in the config file,
    /// and returns them as a metadata pointers.
    /// This should only be called once.
    std::map<std::string, metadataPool*> build_pools();

private:
    void build_from_tree(std::map<std::string, metadataPool*>& pools,
                         const nlohmann::json& config_tree, const std::string& path);

    /// creates a new pool for the given metadata name and location
    metadataPool* new_pool(const std::string& pool_type, const std::string& location) {
        INFO_NON_OO("Creating metadata pool of type: {:s}, at config tree path: {:s}", pool_type,
                    location);

        uint32_t num_metadata_objects = config.get<uint32_t>(location, "num_metadata_objects");

        return create_metadata_pool(num_metadata_objects,
                                    metadataRegistry::kotekan_reg_retrieve(pool_type)->size(),
                                    location.c_str());
    }

    Config& config;
};

} // namespace kotekan

#endif /* METADATA_FACTORY_HPP */
