using Unitary, Test, LinearAlgebra, Flux, Statistics
using Unitary: SVDDense
using Flux.Tracker: Params, gradient
using Flux.Optimise: train!
using Plots
plotly()


function six_gaussians(l, ρ = 8)
	x = reduce(hcat,[randn(2,l) .+ ρ .* [cos(ϕ), sin(ϕ)] for ϕ in 0:45:360])
	# x = x[:,randperm(size(x,2))]
	x = Float32.(x)
end

function single_gaussians(l, c =1)
	Σ = [1.0 c; c 10.0]
	Float32.(cholesky(Σ).U * randn(2, l))
end


x = single_gaussians(100);
xtst = single_gaussians(100)

log_normal(x) = - sum(x.^2, dims=1) / 2 .- size(x,1)*log(2π) / 2

function lkl(m, x)
	x, l = m((x,0))
	log_normal(x) .+ l
end

m = Chain(SVDDense(selu), SVDDense(selu), SVDDense(selu), SVDDense(selu), SVDDense(selu), SVDDense(selu),SVDDense(selu), SVDDense(selu), SVDDense(selu), SVDDense(selu), SVDDense(selu), SVDDense(selu), SVDDense(identity))
opt = ADAM()
ps = params(m)
Flux.train!(x -> -mean(lkl(m, x)), ps, repeatedly(() -> (x,), 1000), opt; cb = () -> @show mean(lkl(m,x)) )

o = Flux.data(m(x))
scatter(o[1,:], o[2,:], title = "projected to latent")
scatter(x[1,:], x[2,:], title = "original data")
z = randn(2,1000)
c = 1
zz = z[:, maximum(abs.(z), dims = 1)[:] .< c]
o = Flux.data(inv(m)(zz))
scatter(o[1,:], o[2,:], title = "generated data")
