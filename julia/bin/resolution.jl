using PrettyTables
using TypedTables

# All quantities in SI units

const c₀ = 299_792_458          # speed of light in vacuum [m/s]

struct Telescope
    ndishes::Int
    separation::Float64
end

function angular_resolution(telescope::Telescope, frequency)
    baseline = telescope.ndishes * telescope.separation
    lambda = c₀ / frequency
    resolution = lambda / baseline
    return resolution
end

function field_of_view(telescope::Telescope, frequency)
    baseline = telescope.separation
    lambda = c₀ / frequency
    resolution = lambda / baseline
    return resolution
end

function show_telescope(
    name::AbstractString, telescope_ew::Telescope, telescope_ns::Telescope, frequency_lo::Real, frequency_hi::Real
)
    println("$name:")
    data = Table(;
        quantity=["angular resolution", "angular resolution", "field of view", "field of view"],
        frequencies=map(f -> "$(round(Int, f / 1e6)) MHz", [frequency_lo, frequency_hi, frequency_lo, frequency_hi]),
        ew_mrad=map(
            x -> "$(round(x * 1000; digits=1)) mrad",
            [
                angular_resolution(telescope_ew, frequency_lo),
                angular_resolution(telescope_ew, frequency_hi),
                field_of_view(telescope_ew, frequency_lo),
                field_of_view(telescope_ew, frequency_hi),
            ],
        ),
        ew_deg=map(
            x -> "$(round(x * 180 / pi; digits=2))°",
            [
                angular_resolution(telescope_ew, frequency_lo),
                angular_resolution(telescope_ew, frequency_hi),
                field_of_view(telescope_ew, frequency_lo),
                field_of_view(telescope_ew, frequency_hi),
            ],
        ),
        ns_mrad=map(
            x -> "$(round(x * 1000; digits=1)) mrad",
            [
                angular_resolution(telescope_ns, frequency_lo),
                angular_resolution(telescope_ns, frequency_hi),
                field_of_view(telescope_ns, frequency_lo),
                field_of_view(telescope_ns, frequency_hi),
            ],
        ),
        ns_deg=map(
            x -> "$(round(x * 180 / pi; digits=2))°",
            [
                angular_resolution(telescope_ns, frequency_lo),
                angular_resolution(telescope_ns, frequency_hi),
                field_of_view(telescope_ns, frequency_lo),
                field_of_view(telescope_ns, frequency_hi),
            ],
        ),
    )
    pretty_table(data; header=["quantity", "frequency", "east-west", "east-west", "north-south", "north-south"], tf=tf_borderless)
    println()
    return nothing
end

function main()
    show_telescope("Pathfinder", Telescope(11, 6.3), Telescope(6, 8.5), 300e6, 1500e6)
    show_telescope("CHORD", Telescope(22, 6.3), Telescope(24, 8.5), 300e6, 1500e6)
    show_telescope("CHIME", Telescope(4, 20.0), Telescope(256, 0.390625), 400e6, 800e6)
    show_telescope("HIRAX", Telescope(16, 6.5), Telescope(16, 8.5), 400e6, 800e6)

    return nothing
end
