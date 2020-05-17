using Unitary, Test, LinearAlgebra, Flux
using Unitary: UnitaryHouseholder, HH_t
using FiniteDifferences
using Zygote: gradient


function HH_reflection(y::Vector)
	t = HH_t(y)
	I - t * y * y'
end

function HH_mul(Y::AbstractMatrix)
	U = I
	for i = 1:size(Y)[2]
		U = U * HH_reflection(Y[:, i])
	end
	U
end

@testset "Conversion of UnitaryHouseholder to matrix" begin
	Y = randn(5, 5)
	a = UnitaryHouseholder(Y)
	Y = LowerTriangular(Y)
	@test Matrix(a) ≈ HH_mul(Y)
	@test Matrix(transpose(a)) ≈ HH_mul(Y[:, end:-1:1])
end

@testset "Multiplication" begin
	a = UnitaryHouseholder(5)
	x = randn(Float32, 5, 8)
	@test a * x ≈ HH_mul(a.Y) * x
	@test transpose(a) * x ≈ HH_mul(a.Y)' * x
	xt = rand(Float32, 8, 5)
	@test xt * a ≈ xt * HH_mul(a.Y)
	@test xt * transpose(a) ≈ xt * HH_mul(a.Y)'
end

@testset "Test partial derivative of single reflection" begin
	y = randn(5)
	for i = 1:5
		δy = zeros(5)
		δy[i] = 10^(-6)
		num = (HH_reflection(y+δy)-HH_reflection(y))*10^6
		@test isapprox(num, Unitary.pdiff_reflect(y, i); atol = 10^(-5))
	end
end

@testset "Test gradient functions" begin
	cfdm = central_fdm(5, 1)
	Y = Matrix(LowerTriangular(rand(5, 5)))
	x = rand(5, 5)
	U = Matrix(UnitaryHouseholder(Y))
	Δ = ones(size(Y*x))
	n = 5

	o = Unitary.mulax(Y, x, false, n)
	ot = Unitary.mulax(Y, x, true, n)
	@test grad(cfdm, Y -> sum(UnitaryHouseholder(Y) * x), Y)[1] ≈
	Unitary.grad_mulax(Δ, Y, false, o)[1]
	@test grad(cfdm, Y -> sum(transpose(UnitaryHouseholder(Y)) * x), Y)[1] ≈
	Unitary.grad_mulax(Δ, Y, true, ot)[1]

	@test grad(cfdm, x -> sum(UnitaryHouseholder(Y) * x), x)[1] ≈
	Unitary.grad_mulax(Δ, Y, false, o)[2]
	@test grad(cfdm, x -> sum(transpose(UnitaryHouseholder(Y)) * x), x)[1] ≈
	Unitary.grad_mulax(Δ, Y, true, ot)[2]

	o = Unitary.mulxa(Y, x, false, n)
	ot = Unitary.mulxa(Y, x, true, n)
	@test grad(cfdm, Y -> sum(x * UnitaryHouseholder(Y)), Y)[1] ≈
	Unitary.grad_mulxa(Δ, Y, false, o)[1]
	@test grad(cfdm, Y -> sum(x * transpose(UnitaryHouseholder(Y))), Y)[1] ≈
	Unitary.grad_mulxa(Δ, Y, true, ot)[1]

	@test grad(cfdm, x -> sum(x * UnitaryHouseholder(Y)), x)[1] ≈
	Unitary.grad_mulxa(Δ, Y, false, o)[2]
	@test grad(cfdm, x -> sum(x * transpose(UnitaryHouseholder(Y))), x)[1] ≈
	Unitary.grad_mulxa(Δ, Y, true, o)[2]
end

@testset "Testing integration with Flux" begin
	Y = Matrix(LowerTriangular(rand(5, 5)))
	x = rand(5, 5)
	U = UnitaryHouseholder(Y)
	ps = Flux.params(U)
	cfdm = central_fdm(5, 1)

	@test gradient(() -> sum(sin.(U * x)), ps)[U.Y] ≈
	grad(cfdm, Y -> sum(sin.(UnitaryHouseholder(Y) * x)), Y)[1]
	@test gradient(x -> sum(sin.(U * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(UnitaryHouseholder(Y) * x)), x)[1]

	@test gradient(() -> sum(sin.(x * U)), ps)[U.Y] ≈
	grad(cfdm, Y -> sum(sin.(x * UnitaryHouseholder(Y))), Y)[1]
	@test gradient(x -> sum(sin.(x * U)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(x * UnitaryHouseholder(Y))), x)[1]

	U = transpose(U)
	@test gradient(() -> sum(sin.(U * x)), ps)[U.Y] ≈
	grad(cfdm, Y -> sum(sin.(transpose(UnitaryHouseholder(Y)) * x)), Y)[1]
	@test gradient(x -> sum(sin.(U * x)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(transpose(UnitaryHouseholder(Y)) * x)), x)[1]

	@test gradient(() -> sum(sin.(x * U)), ps)[U.Y] ≈
	grad(cfdm, Y -> sum(sin.(x * transpose(UnitaryHouseholder(Y)))), Y)[1]
	@test gradient(x -> sum(sin.(x * U)), x)[1] ≈
	grad(cfdm, x -> sum(sin.(x * transpose(UnitaryHouseholder(Y)))), x)[1]
end
