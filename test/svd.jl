using FiniteDifferences, LinearAlgebra, Unitary, Test
using Unitary: Svd

@testset "Test basic Svd functions" begin
	a = Svd(5, :givens)
	@test isapprox(Matrix(transpose(a)), transpose(Matrix(a)), atol = 1e-4)
	@test isapprox(Matrix(inv(a)), inv(Matrix(a)), atol = 1e-4)
	@test Matrix(a) ≈ Matrix(a.u) * Matrix(a.d) * Matrix(a.v) 
	@test Unitary._logabsdet(a) ≈ log(abs(det(Matrix(a))))
	@test size(a) == (5, 5)
	a = Svd(5, :householder)
	@test isapprox(Matrix(transpose(a)), transpose(Matrix(a)), atol = 1e-4)
	@test isapprox(Matrix(inv(a)), inv(Matrix(a)), atol = 1e-4)
	@test Matrix(a) ≈ Matrix(a.u) * Matrix(a.d) * Matrix(a.v) 
	@test Unitary._logabsdet(a) ≈ log(abs(det(Matrix(a))))
	@test size(a) == (5, 5)
end
