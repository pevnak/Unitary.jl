using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryMatrix, _∇mulax, _mulax, _mulatx, _mulxa, _mulxat, SVDDense
using Flux.Tracker: Params, gradient, ngradient

@testset "Testing multiplication and transposed" begin
	a = UnitaryMatrix([1])
	ad = Matrix(a)

	for x in [randn(2),rand(2,10)]
		@test a * x ≈ ad * x
		@test transpose(a) * x ≈ transpose(ad) * x
		@test transpose(x) * transpose(a) ≈ transpose(x) * transpose(ad)
		@test transpose(x) * a ≈ transpose(x) * ad
	end

	for x in [rand(1,2), rand(10, 2)];
		@test x * a ≈ x * ad
		@test x * transpose(a) ≈ x * transpose(ad)
		@test transpose(a) * transpose(x) ≈ transpose(ad) * transpose(x)
		@test a * transpose(x) ≈ ad * transpose(x)
	end
end


@testset "Testing integration with Flux" begin
	for x in [randn(2), randn(2, 10), transpose(randn(10, 2)), transpose(randn(1, 2))]
		θ = [1.0]
		pθ = param(θ)
		a = UnitaryMatrix(pθ)
		px = param(x)
		ps = Params(params(a))
		push!(ps, px)

		#testing gradient of a * x with respect to x and parameters of a
		grads = Flux.Tracker.gradient(() -> sum(sin.(a * px)), ps)
		∇θ, ∇x = Flux.data(grads[pθ]), Flux.data(grads[px])
		@test isapprox(∇x, ngradient(x -> sum(sin.(Flux.data(a) * x)), x)[1], atol = 1e-6)
		@test isapprox(∇θ, ngradient(θ -> sum(sin.(_mulax(θ, x))), θ)[1], atol = 1e-6)

		#testing gradient of transpose(a) * x with respect to x and parameters of a
		at = transpose(a);
		grads = Flux.Tracker.gradient(() -> sum(sin.(at * px)), ps)
		∇θ, ∇x = Flux.data(grads[pθ]), Flux.data(grads[px])
		@test isapprox(∇x, ngradient(x -> sum(sin.(Flux.data(at) * x)), x)[1], atol = 1e-6)
		@test isapprox(∇θ, ngradient(θ -> sum(sin.(_mulatx(θ, x))), θ)[1], atol = 1e-6)
	end

	for x in [rand(10, 2), rand(1, 2), transpose(rand(2,10)), transpose(rand(2)), transpose(rand(2,1))]
		px = param(x)
		ps = Params(params(a))
		push!(ps, px)
		#testing gradient of a * x with respect to x and parameters of a
		grads = Flux.Tracker.gradient(() -> sum(sin.(px * a)), ps)
		∇θ, ∇x = Flux.data(grads[pθ]), Flux.data(grads[px])
		@test isapprox(∇x, ngradient(x -> sum(sin.(x * Flux.data(a) )), x)[1], atol = 1e-6)
		@test isapprox(∇θ, ngradient(θ -> sum(sin.(_mulxa(x, θ))), θ)[1], atol = 1e-6)

		#testing gradient of transpose(a) * x with respect to x and parameters of a
		at = transpose(a);
		grads = Flux.Tracker.gradient(() -> sum(sin.(px * at)), ps)
		∇θ, ∇x = Flux.data(grads[pθ]), Flux.data(grads[px])
		@test isapprox(∇x, ngradient(x -> sum(sin.(x * Flux.data(at) )), x)[1], atol = 1e-6)
		@test isapprox(∇θ, ngradient(θ -> sum(sin.(_mulxat(x, θ))), θ)[1], atol = 1e-6)
	end
end

@testset "Testing calculation of the Jacobian" begin
	jacobian(f, x) = vcat([transpose(Flux.data(gradient(x -> f(x)[i], x)[1])) for i in 1:length(x)]...)
	for σ in [identity, selu]
		m = SVDDense(σ)

		x = randn(2,1)
		@test log(abs(det(jacobian(m, x)))) ≈ Flux.data(m((x,0))[2])[1]
	end
end

@testset "Gradient of the likelihood" begin
	m = SVDDense(selu)
	function lkl(m, x)
		x, l = m((x,0))
		exp.(- sum(x.^2, dims = 1)) .+ l
	end
	x = randn(2,10)
	@test isapprox(gradient(x -> sum(lkl(m, x)), x)[1], ngradient(x -> Flux.data(sum(lkl(m, x))), x)[1], atol = 1e-6)
	m = Chain(SVDDense(selu), SVDDense(selu), SVDDense(identity))
	@test isapprox(gradient(x -> sum(lkl(m, x)), x)[1], ngradient(x -> Flux.data(sum(lkl(m, x))), x)[1], atol = 1e-5)
end