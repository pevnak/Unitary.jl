using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryMatrix, _∇mulax, _mulax, _mulatx
using Flux.Tracker: Params

@testset "Testing multiplication and transposed" begin
	a = UnitaryMatrix([1])
	ad = Matrix(a)

	# for x in [randn(10),rand(2,10)]
	x = rand(2,10)
		@test a * x ≈ ad * x
		@test transpose(a) * x ≈ transpose(ad) * x
		@test transpose(x) * transpose(a) ≈ transpose(x) * transpose(ad)
		@test transpose(x) * a ≈ transpose(x) * ad
	# end

	# for x in [rand(1,2), rand(10, 2)];
	x = rand(10,  2)
		@test x * a ≈ x * ad
		@test x * transpose(a) ≈ x * transpose(ad)
		@test transpose(a) * transpose(x) ≈ transpose(ad) * transpose(x)
		@test a * transpose(x) ≈ ad * transpose(x)
	# end
end


@testset "Integration with Flux" begin
	θ = [1.0]
	a = UnitaryMatrix(param(θ))
	x = param(rand(2,10))
	ps = Params(params(a))
	push!(ps, x)

	grads = Flux.Tracker.gradient(() -> sum(sin.(a * x)), ps)
	∇θ = Flux.data(grads[a.θ])
	∇x = Flux.data(grads[x])

	a, x = Flux.data(a), Flux.data(x)
	isapprox(∇x, Flux.Tracker.ngradient(x -> sum(sin.(a * x)), x)[1], atol = 1e-6)
	isapprox(∇θ, Flux.Tracker.ngradient(θ -> sum(sin.(_mulax(θ, x))), θ)[1], atol = 1e-6)

	gradtest(a -> a*x, a)
	isapprox(Flux.Tracker.ngradient(θ -> sum(Unitary._mulatx(x, θ)),θ)[1], _∇mulax(θ, ones(size(x)), x), atol = 1e-6)

	@test a * x ≈ ad * x
	x = rand(10, 2)
	@test x * a ≈ x * ad
end

