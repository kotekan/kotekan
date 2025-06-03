#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <DataType.hpp>         // for DataType
#include <Symbol.hpp>           // for Symbol
#include <ext/new_allocator.h>  // for ptrdiff_t
#include <array>                // for array
#include <cassert>              // for assert
#include <functional>           // for function
#include <initializer_list>     // for initializer_list
#include <iostream>             // for ptrdiff_t, size_t, ostream
#include <utility>              // for pair
#include <vector>               // for vector

namespace kotekan {

namespace {

// Helper function to create a `std::array` from an initializer list.
// The initializer list must have the correct size.
template<std::size_t D, typename T>
std::array<T, D> make_array(std::initializer_list<T> values) {
    assert(values.size() == D);
    std::array<T, D> result;
    for (std::size_t d = 0; d < D; ++d)
        result[d] = values.begin()[d];
    return result;
}

// Helper function to create a pair of `std::array`s from an
// initializer list of pairs. (This transposes the input.) The
// initializer list must have the correct size.
template<std::size_t D, typename T, typename U>
std::pair<std::array<T, D>, std::array<U, D>>
make_arrays_from_pairs(std::initializer_list<std::pair<T, U>> values) {
    assert(values.size() == D);
    std::pair<std::array<T, D>, std::array<U, D>> result;
    for (std::size_t d = 0; d < D; ++d)
        result.first[d] = values.begin()[d].first;
    for (std::size_t d = 0; d < D; ++d)
        result.second[d] = values.begin()[d].second;
    return result;
}

} // namespace

// MISSING:
// - get/set chordDatatype
// - attach to buffer (set buffer name)
// - attach to ringbuffer (set buffer name)
// - set ringbuffer dimension name
// - set ringbuffer capacity
// - get ringbuffer stride
// - buffer depth?
// - wrapper functions for waiting for data, writing data
// - wrapper functions for ringbuffer waiting, selecting, writing
// - check buffer length when attaching
// - define extra class for ringbuffer management?
// - register producers, consumers?
// - register Graphviz entries
// - what about metadata? (read? check? define?)
// - I/O methods
// - yaml interface?
// - should the rank remain a compile-time constant?
// - new type `NDBuffer`?
// - add functions to allocate/deallocate buffer?

// A `GenericNDArray` is similar to a numpy array. Neither the type
// nor the rank are known at compile time. This is convenient e.g.
// when reading data from a file, but it prevents efficient operations
// on array elements.
class GenericNDArray {
public:
    virtual ~GenericNDArray() {}
    // The value (element) type
    virtual DataType get_value_type() const = 0;
    // The number of bytes for each value (element)
    virtual std::size_t get_value_type_size() const = 0;
    // The rank (number of dimensions)
    virtual std::size_t get_rank() const = 0;
    // Pointer to the array data
    virtual const void* get_data() const = 0;
    virtual void* get_data() = 0;
    // The array extents (array shape)
    virtual std::vector<std::ptrdiff_t> get_extents() const = 0;
    virtual std::ptrdiff_t get_extent(std::size_t d) const = 0;
    // Is the array empty?
    virtual bool get_empty() const = 0;
    // The array size. (This is the product of the extents.)
    virtual std::ptrdiff_t get_size() const = 0;
    // The names of the array dimensions
    virtual std::vector<Symbol> get_dimnames() const = 0;
    virtual Symbol get_dimname(std::size_t d) const = 0;
    // The array strides. Strides can have any value (including negative).
    virtual std::vector<std::ptrdiff_t> get_strides() const = 0;
    virtual std::ptrdiff_t get_stride(std::size_t d) const = 0;

    // Output the array metadata, useful for logging or debugging
    void output_metadata(std::ostream& os) const;
};

// A `NDArray<T,D>` is a `D`-dimensional array of type `T`. Different
// from `GenericNDArray`, its elements can be accessed efficiently. If
// you are writing C++ code that processes multi-dimensional arrays
// then this class is a fine choice.
template<typename T, std::size_t D>
class NDArray : public GenericNDArray {
public:
    // Value (element) type
    using value_type = T;
    // Rank (number of dimensions)
    constexpr static std::size_t rank = D;
    constexpr static std::size_t value_type_size = sizeof(T);

private:
    // Pointer to the array data
    T* m_data;

    // Has a target if we own the data and need to deallocate them
    std::function<void()> m_cleanup;

    // Extents (shape)
    std::array<std::ptrdiff_t, D> m_extents;
    // Size (number of elements)
    std::ptrdiff_t m_size;

    // Strides
    std::array<std::ptrdiff_t, D> m_strides;

    // Dimension names
    std::array<Symbol, D> m_dimnames;

public:
    // No default constructor
    NDArray() = delete;

    // No copy constructors
    NDArray(const NDArray&) = delete;
    NDArray& operator=(const NDArray&) = delete;

    // Move constructors
    NDArray(NDArray&&) = default;
    NDArray& operator=(NDArray&&) = default;

    // Construct from extents and dimension names
    template<typename I = std::ptrdiff_t>
    NDArray(const std::array<I, D>& extents, const std::array<Symbol, D>& dimnames) {
        for (std::size_t d = 0; d < D; ++d)
            m_extents[d] = extents[d];
        for (std::size_t d = 0; d < D; ++d)
            assert(m_extents[d] >= 0);

        for (std::size_t d = 0; d < D; ++d)
            m_dimnames[d] = dimnames[d];
        for (std::size_t d = 0; d < D; ++d)
            for (std::size_t d1 = 0; d1 < d; ++d1)
                assert(m_dimnames[d] != m_dimnames[d1]);

        m_size = 1;
        for (std::size_t d = 0; d < D; ++d)
            m_size *= m_extents[d];
        for (std::size_t d = D; d > 0; --d)
            m_strides[d - 1] = d == D ? 1 : m_strides[d] * m_extents[d];

        // We allocate the array without initializing its elements
        m_data = new T[m_size];
        m_cleanup = [&]() { delete[] m_data; };
    }
    template<typename I = std::ptrdiff_t>
    NDArray(std::initializer_list<I> extents, std::initializer_list<Symbol> dimnames) :
        NDArray(make_array<D>(extents), make_array<D>(dimnames)) {}
    template<typename I = std::ptrdiff_t>
    NDArray(const std::pair<std::array<Symbol, D>, std::array<I, D>>& dimnames_extents) :
        NDArray(dimnames_extents.second, dimnames_extents.first) {}
    template<typename I = std::ptrdiff_t>
    NDArray(std::initializer_list<std::pair<Symbol, I>> dimnames_extents) :
        NDArray(make_arrays_from_pairs<D>(dimnames_extents)) {}

    virtual ~NDArray() {
        if (m_cleanup)
            m_cleanup();
    }

    // Get a pointer to the first element
    const T* data() const {
        return m_data;
    }
    T* data() {
        return m_data;
    }

    // The array extents (array shape)
    std::array<std::ptrdiff_t, D> extents() const {
        return m_extents;
    }
    std::ptrdiff_t extent(std::size_t d) const {
        assert(d < D);
        return m_extents[d];
    }

    // Is the array empty?
    bool empty() const {
        return size() == 0;
    }

    // The array size. (This is the product of the extents.)
    std::ptrdiff_t size() const {
        return m_size;
    }

    // The names of the array dimensions
    std::array<Symbol, D> dimnames() const {
        return m_dimnames;
    }
    Symbol dimname(std::size_t d) const {
        assert(d < D);
        return m_dimnames[d];
    }

    // The array strides. Strides can have any value (including negative).
    std::array<std::ptrdiff_t, D> strides() const {
        return m_strides;
    }
    std::ptrdiff_t stride(std::size_t d) const {
        assert(d < D);
        return m_strides[d];
    }

    // Convert an array index to an offset, using the strides
    template<typename I = std::ptrdiff_t>
    std::ptrdiff_t index2offset(const std::array<I, D>& ind) const {
        for (std::size_t d = 0; d < D; ++d)
            assert(ind[d] >= 0 && ind[d] < m_extents[d]);
        std::ptrdiff_t off = 0;
        for (std::size_t d = 0; d < D; ++d)
            off += ind[d] * m_strides[d];
        return off;
    }

    // Access an array element by index
    template<typename I = std::ptrdiff_t>
    const T& operator()(const std::array<I, D>& ind) const {
        return m_data[index2offset(ind)];
    }
    template<typename I = std::ptrdiff_t>
    T& operator()(const std::array<I, D>& ind) {
        return m_data[index2offset(ind)];
    }

    // A const iterator
    class ConstIterator {
        const NDArray<T, D>* m_parent;
        std::array<std::ptrdiff_t, D> m_indices;
        ConstIterator() = delete;
        ConstIterator& operator++() {
            for (std::size_t d = 0; d < D; ++d) {
                if (d == D - 1 || m_indices[d] < m_parent->extent(d)) {
                    ++m_indices[d];
                    break;
                } else {
                    m_indices[d] = 0;
                }
            }
            return *this;
        }
        ConstIterator& operator++(int) {
            return ++*this;
        }
        friend bool operator==(const ConstIterator& iter1, const ConstIterator& iter2) {
            return iter1.m_indices == iter2.m_indices;
        }
        friend bool operator!=(const ConstIterator& iter1, const ConstIterator& iter2) {
            return !(iter1 == iter2);
        }
        const T& operator*() const {
            return (*m_parent)(m_indices);
        }
    };
    ConstIterator cbegin() const {
        std::array<std::ptrdiff_t, D> first;
        for (std::size_t d = 0; d < D; ++d)
            first[d] = 0;
        return ConstIterator{this, first};
    }
    ConstIterator cend() const {
        std::array<std::ptrdiff_t, D> past_last;
        for (std::size_t d = 0; d < D; ++d)
            past_last[d] = d < D - 1 ? 0 : extent(d);
        return ConstIterator{this, past_last};
    }
    ConstIterator begin() const {
        return cbegin();
    }
    ConstIterator end() const {
        return cend();
    }

    // TODO: Add begin, end iterators?

    // For GenericNDArray

    DataType get_value_type() const override {
        return GetDataType_v<value_type>;
    }
    std::size_t get_value_type_size() const override {
        return value_type_size;
    }
    std::size_t get_rank() const override {
        return rank;
    }
    const void* get_data() const override {
        return data();
    }
    void* get_data() override {
        return data();
    }
    std::vector<std::ptrdiff_t> get_extents() const override {
        std::vector<std::ptrdiff_t> result(D);
        for (std::size_t d = 0; d < D; ++d)
            result.at(d) = extents()[d];
        return result;
    }
    std::ptrdiff_t get_extent(std::size_t d) const override {
        return extent(d);
    }
    bool get_empty() const override {
        return empty();
    }
    std::ptrdiff_t get_size() const override {
        return size();
    }
    std::vector<Symbol> get_dimnames() const override {
        std::vector<Symbol> result(D);
        for (std::size_t d = 0; d < D; ++d)
            result.at(d) = dimnames()[d];
        return result;
    }
    Symbol get_dimname(std::size_t d) const override {
        return dimname(d);
    }
    std::vector<std::ptrdiff_t> get_strides() const override {
        std::vector<std::ptrdiff_t> result(D);
        for (std::size_t d = 0; d < D; ++d)
            result.at(d) = strides()[d];
        return result;
    }
    std::ptrdiff_t get_stride(std::size_t d) const override {
        return stride(d);
    }
};

} // namespace kotekan

#endif // #ifndef NDARRAY_HPP
