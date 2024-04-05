/*****************************************
@file
@brief Code for using the VisFrameView formatted data.
- VisMetadata
- VisFrameView
*****************************************/
#ifndef N2BUFFER_HPP
#define N2BUFFER_HPP

#include "Config.hpp"        // for Config
#include "FrameView.hpp"     // for FrameView
#include "Telescope.hpp"     // for freq_id_t
#include "buffer.hpp"        // for Buffer
#include "N2Util.hpp"        // for cfloat
#include "N2Metadata.hpp"    // for N2Metadata

#include "gsl-lite.hpp"      // for span

#include <map>               // for map
#include <utility>           // for pair
#include <iostream>          // for cout

/**
 * @brief The fields within the N2FrameView.
 *
 * Use this enum to refer to the fields.
 **/
enum class N2Field { vis, weight, flags, eval, evec, erms, gain };

/**
 * @class N2FrameView
 * @brief Provide a structured view of a visn N2k-pipeline visibility buffer.
 *
 * This class inherits from the FrameView base class and sets up a view on a visibility buffer with
 * the ability to interact with the data and metadata.
 *
 **/
class N2FrameView :
    public FrameView {

public:
    std::shared_ptr<N2Metadata> const _metadata;

    std::map<N2Field, std::pair<size_t, size_t>> frame_layout;

    /// View of the visibility data.
    gsl::span<N2::cfloat> vis;
    /// View of the weight data.
    gsl::span<float> weight;
    /// View of the input flags
    gsl::span<float> flags;
    /// View of the eigenvalues.
    gsl::span<float> eval;
    /// View of the eigenvectors (packed as ev,feed).
    gsl::span<N2::cfloat> evec;
    /// The RMS of residual visibilities
    float& erms;
    /// View of the applied gains
    gsl::span<N2::cfloat> gain;

    /**
     * @brief The sizes of the fields in the N2FrameView.
     */
    static std::vector<std::pair<N2Field, size_t>> get_field_sizes(uint32_t num_elements, uint32_t num_ev) {

        size_t num_prod = N2::get_num_prod(num_elements);

        std::vector<std::pair<N2Field, size_t>> field_sizes;
        field_sizes.push_back( { N2Field::vis, sizeof(N2::cfloat) * num_prod } );
        std::cout << " ---- " << (int) field_sizes[0].first << " " << (int) field_sizes[0].second << std::endl;
        field_sizes.push_back( { N2Field::weight, sizeof(float) * num_prod} );
        std::cout << " ---- " << (int) field_sizes[1].first << " " << (int) field_sizes[1].second << std::endl;
        field_sizes.push_back( { N2Field::flags, sizeof(float) * num_elements} );
        field_sizes.push_back( { N2Field::eval, sizeof(float) * num_ev} );
        field_sizes.push_back( { N2Field::evec, sizeof(N2::cfloat) * num_ev * num_elements} );
        field_sizes.push_back( { N2Field::erms, sizeof(float) * 1} );
        field_sizes.push_back( { N2Field::gain, sizeof(N2::cfloat) * num_elements} );

        return field_sizes;
    }

    /**
     * @brief The layout of the fields within the N2FrameView.
     * 
     * @return A map of the field to the { start, end } of the field in the frame.
     **/
    static std::map<N2Field, std::pair<size_t, size_t>> get_frame_layout(uint32_t num_elements, uint32_t num_ev)
    {
        std::map<N2Field, std::pair<size_t, size_t>> frame_layout;
        std::vector<std::pair<N2Field, size_t>> field_sizes = get_field_sizes(num_elements, num_ev);
        
        // build the layout
        size_t offset = 0;
        for (const std::pair<N2Field, size_t> & field : field_sizes) {
            frame_layout[field.first] = std::make_pair(offset, offset + field.second);
            std::cout << " -- Frame " << (int) field.first << ": {"  << frame_layout[field.first].first <<  ", " << (int) frame_layout[field.first].second << " }" << std::endl;
            offset += field.second;
        }

        return frame_layout;
    }

    /**
     * @brief Calculate the size of the frame.
     */
    static size_t calculate_frame_size(uint32_t num_elements, uint32_t num_ev) {
        size_t frame_size = get_frame_layout(num_elements, num_ev)[N2Field::gain].second;
        return frame_size;
    }

    static size_t calculate_frame_size(kotekan::Config& config, const std::string& unique_name) {

        const int num_elements = config.get<int>(unique_name, "num_elements");
        const int num_ev = config.get<int>(unique_name, "num_ev");

        return calculate_frame_size(num_elements, num_ev);
    }


    /**
     * @brief Create view without modifying layout.
     *
     * This should be used for viewing already created frames.
     *
     * @param buf      The buffer the frame is in.
     * @param frame_id The id of the frame to read.
     */
    N2FrameView(Buffer* buf, int frame_id) :

        FrameView(buf, frame_id), _metadata(std::static_pointer_cast<N2Metadata>(buf->metadata[id])),
        frame_layout(get_frame_layout(_metadata->num_elements, _metadata->num_ev)),

        vis(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::vis])),
        weight(bind_span<float>(_frame, frame_layout[N2Field::weight])),
        flags(bind_span<float>(_frame, frame_layout[N2Field::flags])),
        eval(bind_span<float>(_frame, frame_layout[N2Field::eval])),
        evec(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::evec])),
        erms(bind_scalar<float>(_frame, frame_layout[N2Field::erms])),
        gain(bind_span<N2::cfloat>(_frame, frame_layout[N2Field::gain])) {

        // assert frame size is correct
        assert(data_size() == buf->frame_size);
    }
    
    size_t data_size() const override;
    void zero_frame() override;
};

#endif
