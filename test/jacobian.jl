using Unitary, Test, LinearAlgebra, Flux, Zygote
using Unitary: SVDDense


@testset "Testing calculation of the Jacobian" begin
	jacobian(f, x) = vcat([transpose(gradient(x -> f(x)[i], x)[1]) for i in 1:length(x)]...)
	for σ in [identity, selu]
		m = SVDDense(2,σ)

		x = randn(2,1)
		@test isapprox(log(abs(det(jacobian(m, x)))), m((x,0))[2][1], atol = 1e-4)
	end
	for σ in [identity, selu]
		m = Chain(SVDDense(2,selu), SVDDense(2,selu), SVDDense(2,identity))

		x = randn(2,1)
		@test isapprox(log(abs(det(jacobian(m, x)))), m((x,0))[2][1], atol = 1e-3)
	end
end

@testset "Gradient of the likelihood" begin
	m = SVDDense(2,selu)
	function lkl(m, x)
		x, l = m((x,0))
		exp.(- sum(x.^2, dims = 1)) .+ l
	end
	x = randn(2,10)
	fdm = central_fdm(5, 1)
	@test isapprox(gradient(x -> sum(lkl(m, x)), x)[1], grad(fdm, x -> sum(lkl(m, x)), x)[1], atol = 1e-5)
	m = Chain(SVDDense(2,selu), SVDDense(2,selu), SVDDense(2,identity))
	@test isapprox(gradient(x -> sum(lkl(m, x)), x)[1], grad(fdm, x -> sum(lkl(m, x)), x)[1], atol = 1e-5)
end
