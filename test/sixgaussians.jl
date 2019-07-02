using Unitary, Test, LinearAlgebra, Flux
using Unitary: SVDDense
using Flux.Tracker: Params, gradient

function six_gaussians(l, ρ = 8)
	x = reduce(hcat,[randn(2,l) .+ ρ .* [cos(ϕ), sin(ϕ)] for ϕ in 0:45:360])
	x = x[:,randperm(size(x,2))]
	x = Float32.(x)
end

x = six_gaussians(100);
xtst = six_gaussians(100)

log_normal(x) = - sum(x.^2, dims=1) / 2 .- size(x,1)*log(2π) / 2

m = Chain(SVDDense(selu), SVDDense(selu), SVDDense(identity))
function lkl(m, x)
	x, l = m((x,0))
	log_normal(x) .+ l
end

