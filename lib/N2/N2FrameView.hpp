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
#include "N2Metadata.hpp"    // for cfloat

#include "gsl-lite.hpp"      // for span

#include <map>               // for map
#include <utility>           // for pair

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
    gsl::span<cfloat> vis;
    /// View of the weight data.
    gsl::span<float> weight;
    /// View of the input flags
    gsl::span<float> flags;
    /// View of the eigenvalues.
    gsl::span<float> eval;
    /// View of the eigenvectors (packed as ev,feed).
    gsl::span<cfloat> evec;
    /// The RMS of residual visibilities
    float& erms;
    /// View of the applied gains
    gsl::span<cfloat> gain;

    /**
     * @brief The layout of the fields within the N2FrameView.
     **/
    std::map<N2Field, std::pair<size_t, size_t>> get_frame_layout(uint32_t num_elements, uint32_t num_ev)
    {
        uint32_t num_prod = num_elements * (num_elements + 1) / 2;

        std::map<N2Field, std::pair<size_t, size_t>> frame_layout;

        frame_layout[ N2Field::vis ] = { 0, sizeof(cfloat) * num_prod };
        frame_layout[ N2Field::weight ] = { frame_layout[ N2Field::vis ].second, sizeof(float) * num_prod };
        frame_layout[ N2Field::flags ] = { frame_layout[ N2Field::weight ].second, sizeof(float) * num_elements };
        frame_layout[ N2Field::eval ] = { frame_layout[ N2Field::flags ].second, sizeof(float) * num_ev };
        frame_layout[ N2Field::evec ] = { frame_layout[ N2Field::eval ].second, sizeof(cfloat) * num_ev * num_elements };
        frame_layout[ N2Field::erms ] = { frame_layout[ N2Field::evec ].second, sizeof(float) * 1 };
        frame_layout[ N2Field::gain ] = { frame_layout[ N2Field::erms ].second, sizeof(cfloat) * num_elements };

        return frame_layout;
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

        vis(bind_span<cfloat>(_frame, frame_layout[N2Field::vis])),
        weight(bind_span<float>(_frame, frame_layout[N2Field::weight])),
        flags(bind_span<float>(_frame, frame_layout[N2Field::flags])),
        eval(bind_span<float>(_frame, frame_layout[N2Field::eval])),
        evec(bind_span<cfloat>(_frame, frame_layout[N2Field::evec])),
        erms(bind_scalar<float>(_frame, frame_layout[N2Field::erms])),
        gain(bind_span<cfloat>(_frame, frame_layout[N2Field::gain])) {

        // ASSERT frame size is big enough
        
    }

    size_t data_size() const override;
    void zero_frame() override;
};

#endif
