using Unitary, Test, LinearAlgebra, Flux, Zygote
using Unitary: UnitaryMatrix, _∇mulax, _mulax, _mulatx, _mulxa, _mulxat, SVDDense

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

@testset "Testing multiplication and transposition" begin
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
		a = UnitaryMatrix(θ)
		ps = Params([θ, x])

		#testing gradient of a * x with respect to x and parameters of a
		grads = gradient(() -> sum(sin.(a * x)), ps)
		∇θ, ∇x = grads[θ], grads[x]
		@test isapprox(∇x, ngradient(x -> sum(sin.(a * x)), x)[1], atol = 1e-6)
		@test isapprox(∇θ, ngradient(θ -> sum(sin.(_mulax(θ, x))), θ)[1], atol = 1e-6)

		#testing gradient of transpose(a) * x with respect to x and parameters of a
		grads = gradient(() -> sum(sin.(transpose(a) * x)), ps)
		∇θ, ∇x = grads[θ], grads[x]
		@test isapprox(∇x, ngradient(x -> sum(sin.(transpose(a) * x)), x)[1], atol = 1e-6)
		@test isapprox(∇θ, ngradient(θ -> sum(sin.(_mulatx(θ, x))), θ)[1], atol = 1e-6)
	end

	for x in [rand(10, 2), rand(1, 2), transpose(rand(2,10)), transpose(rand(2)), transpose(rand(2,1))]
		θ = [1.0]
		a = UnitaryMatrix(θ)
		ps = Params([θ, x])
		#testing gradient of a * x with respect to x and parameters of a
		grads =gradient(() -> sum(sin.(x * a)), ps)
		∇θ, ∇x = grads[θ], grads[x]
		@test isapprox(∇x, ngradient(x -> sum(sin.(x * a)), x)[1], atol = 1e-6)
		@test isapprox(∇θ, ngradient(θ -> sum(sin.(_mulxa(x, θ))), θ)[1], atol = 1e-6)

		#testing gradient of transpose(a) * x with respect to x and parameters of a
		grads = gradient(() -> sum(sin.(x * transpose(a))), ps)
		∇θ, ∇x = grads[θ], grads[x]
		@test isapprox(∇x, ngradient(x -> sum(sin.(x * transpose(a) )), x)[1], atol = 1e-6)
		@test isapprox(∇θ, ngradient(θ -> sum(sin.(_mulxat(x, θ))), θ)[1], atol = 1e-6)
	end
end

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
	@test isapprox(gradient(x -> sum(lkl(m, x)), x)[1], ngradient(x -> sum(lkl(m, x)), x)[1], atol = 1e-6)
	m = Chain(SVDDense(2,selu), SVDDense(2,selu), SVDDense(2,identity))
	@test isapprox(gradient(x -> sum(lkl(m, x)), x)[1], ngradient(x -> sum(lkl(m, x)), x)[1], atol = 1e-5)
end