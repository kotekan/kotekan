#include "../include/pirate/YamlFile.hpp"
#include "../include/pirate/file_utils.hpp"  // file_exists()

#include <sstream>
#include <ksgpu/xassert.hpp>

using namespace std;

namespace pirate {
#if 0
}   // compiler pacifier
#endif


// Helper for constructor.
static YAML::Node load_yaml(const string &filename)
{
    if (!file_exists(filename))
        throw runtime_error("File '" + filename + "' not found");
    return YAML::LoadFile(filename);
}


YamlFile::YamlFile(const string &filename) :
    YamlFile(filename, load_yaml(filename))
{ }


YamlFile::YamlFile(const string &name_, const YAML::Node &node_) :
    name(name_), node(node_)
{ }


YamlFile YamlFile::operator[](const string &k) const
{
    YamlFile child = _get_child(k);

    if (!child.node)
        throw runtime_error(name + ": key '" + k + "' not found");

    return child;
}


YamlFile YamlFile::operator[](long ix) const
{
    this->assert_type_is(Type::Sequence);

    if ((ix < 0) || (ix >= size())) {
        stringstream ss;
        ss << name << ": bad index " << ix << " (size=" << size() << ")";
        throw runtime_error(ss.str());
    }

    stringstream ss;
    ss << name << "[" << ix << "]";

    YamlFile child(ss.str(), node[ix]);
    xassert(child.node);  // should never fail

    return child;
}


bool YamlFile::has_key(const string &k) const
{
    this->assert_type_is(Type::Map);
    return node[k] ? true : false;
}


YamlFile YamlFile::_get_child(const string &k) const
{
    this->assert_type_is(Type::Map);
    
    stringstream ss;
    ss << name << "[" << k << "]";
    
    YamlFile child(ss.str(), node[k]);

    if (child.node)
        this->requested_keys.insert(k);

    return child;    
}


void YamlFile::assert_type_is(Type t) const
{
    if (type() != t) {
        stringstream ss;
        ss << this->name << ": expected " << type_str(t) << ", got " << type_str(type());
        throw runtime_error(ss.str());
    }
}


void YamlFile::check_for_invalid_keys() const
{
    this->assert_type_is(Type::Map);
    vector<string> unrequested_keys;
    
    for (YAML::const_iterator it = node.begin(); it != node.end(); it++) {
        const string &k = it->first.as<string>();
        
        if (requested_keys.find(k) == requested_keys.end())
            unrequested_keys.push_back(k);
    }

    if (unrequested_keys.size() == 0)
        return;

    stringstream ss;
    ss << name << ": the following unexpected key(s) were present: "
       << ksgpu::tuple_str(unrequested_keys);

    throw runtime_error(ss.str());
}


// https://github.com/jbeder/yaml-cpp/blob/master/include/yaml-cpp/node/type.h
string YamlFile::type_str(Type t)
{
    switch (t) {
    case YamlFile::Type::Undefined:
        return "Undefined";
    case YamlFile::Type::Null:
        return "Null";
    case YamlFile::Type::Scalar:
        return "Scalar";
    case YamlFile::Type::Sequence:
        return "Sequence";
    case YamlFile::Type::Map:
        return "Map";
    default:
        return "Unrecognized";
    }
}


}   // namespace pirate
