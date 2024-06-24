include("bin/plot_upchan_voltage.jl")

Emax = vec(maximum(abs ∘ i2c, (@view array_E[:, :, :, :]); dims=(1, 2, 4)));
Emaxind = findmax(Emax)
scatter(Emax[1:20])

Ebarmax = vec(maximum(abs ∘ i2c, (@view array_Ebar[:, :, :, :]); dims=(1, 2, 4)));
Ebarmaxind = findmax(Ebarmax)
scatter(Ebarmax[1:50])

scatterlines(real.(i2c.(array_E[1, 1, Emaxind[2], 1:100])))

include("src/FEngine.jl")
using .FEngine

y = FEngine.upchan(Complex{Float32}.(i2c.(array_E[1, 1, Emaxind[2], :])), 4, 2);
maximum(abs, y; dims=2)

# lines(imag.(i2c.(array_Ebar[1, 1, 12, 1:100])))
# lines(imag.(y[2, 1:100]))
