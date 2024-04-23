#include "N2FrameView.hpp"
#include "buffer.hpp"        // for Buffer
#include "FrameView.hpp"     // for FrameView, bind_span

N2FrameView::N2FrameView(Buffer* buf, int frame_id) :

    FrameView(buf, frame_id), _metadata(std::static_pointer_cast<N2Metadata>(buf->metadata[id])),

    // Set the const refs to the structural metadata
    num_elements(_metadata->num_elements), num_prod(_metadata->num_prod), num_ev(_metadata->num_ev),
    frame_layout(get_frame_layout(_metadata->num_elements, _metadata->num_ev)),

    // Non-structural data
    freq_id(_metadata->freq_id),

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

size_t N2FrameView::data_size() const {
    return calculate_frame_size(_metadata->num_elements, _metadata->num_ev);
}

void N2FrameView::zero_frame() {
    // Fill data with zeros
    std::memset(_frame, 0, data_size());
    erms = 0;

    // Set non-structural metadata
    // time = std::make_tuple(0, timespec{0, 0});
    // n_valid_fpga_samples = 0;

    // // mark frame as empty by ensuring this is 0
    // fpga_seq_length = 0;
    // fpga_seq_total = 0;
}

void N2FrameView::copy_data(N2FrameView frame_to_copy_from, const std::set<N2Field>& skip_members)
{
    // Define some helper methods so we don't need to code up the same checks everywhere
    auto copy_member = [&](N2Field member) { return (skip_members.count(member) == 0); };

    auto check_elements = [&]() {
        if (num_elements != frame_to_copy_from.num_elements) {
            auto msg = fmt::format(fmt("Number of inputs don't match for copy [src={}; dest={}]."),
                                   frame_to_copy_from.num_elements, num_elements);
            throw std::runtime_error(msg);
        }
    };

    auto check_prod = [&]() {
        if (num_elements != frame_to_copy_from.num_elements) {
            auto msg =
                fmt::format(fmt("Number of products don't match for copy [src={}; dest={}]."),
                            frame_to_copy_from.num_prod, num_prod);
            throw std::runtime_error(msg);
        }
    };

    auto check_ev = [&]() {
        if (num_ev != frame_to_copy_from.num_ev) {
            auto msg = fmt::format(fmt("Number of ev don't match for copy [src={}; dest={}]."),
                                   frame_to_copy_from.num_ev, num_ev);
            throw std::runtime_error(msg);
        }
    };

    if (copy_member(N2Field::vis)) {
        check_prod();
        std::copy(frame_to_copy_from.vis.begin(), frame_to_copy_from.vis.end(), vis.begin());
    }

    if (copy_member(N2Field::weight)) {
        check_prod();
        std::copy(frame_to_copy_from.weight.begin(), frame_to_copy_from.weight.end(), weight.begin());
    }


    if (copy_member(N2Field::flags)) {
        check_elements();
        std::copy(frame_to_copy_from.flags.begin(), frame_to_copy_from.flags.end(), flags.begin());
    }

    if (copy_member(N2Field::eval)) {
        check_ev();
        std::copy(frame_to_copy_from.eval.begin(), frame_to_copy_from.eval.end(), eval.begin());
    }

    if (copy_member(N2Field::evec)) {
        check_ev();
        check_elements();
        std::copy(frame_to_copy_from.evec.begin(), frame_to_copy_from.evec.end(), evec.begin());
    }

    if (copy_member(N2Field::erms))
        erms = frame_to_copy_from.erms;

    if (copy_member(N2Field::gain)) {
        check_elements();
        std::copy(frame_to_copy_from.gain.begin(), frame_to_copy_from.gain.end(), gain.begin());
    }
}


