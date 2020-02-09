using Zygote: dropgrad
struct InPlaceUnitaryButterfly{T<:Real,I<:Tuple,N}
	θs::Vector{T}
	θi::NTuple{N,UnitRange{Int64}}
	is::Vector{I}
	js::Vector{I}
	transposed::Int
	n::Int
end

Flux.@functor(InPlaceUnitaryButterfly)
Flux.trainable(m::InPlaceUnitaryButterfly) = [m.θs]

Base.size(a::InPlaceUnitaryButterfly) = (a.n,a.n)
Base.size(a::InPlaceUnitaryButterfly, i::Int) = a.n

Base.eltype(a::InPlaceUnitaryButterfly{T,I}) where {T,I} = T
LinearAlgebra.transpose(a::InPlaceUnitaryButterfly) = InPlaceUnitaryButterfly(a.θs, reverse(a.θi), reverse(a.is), reverse(a.js), -a.transposed, a.n)
Base.inv(a::InPlaceUnitaryButterfly) = transpose(a)
Base.show(io::IO, a::InPlaceUnitaryButterfly) = print(io, "$(a.n)x$(a.n) Unitary with $(size(a.θs,2)) butterfly matrices")
Base.zero(a::InPlaceUnitaryButterfly) = InPlaceUnitaryButterfly(zero(a.θ), a.i, a.j, a.transposed, a.n)

*(a::InPlaceUnitaryButterfly, x::AbstractMatVec) = (@assert size(x,1) == a.n; _buttermulax(a.θs, a.θi, a.is, a.js, a.transposed, x))
*(x::AbstractMatVec, a::InPlaceUnitaryButterfly) = (@assert size(x,2) == a.n; _buttermulxa(x, a.θs, a.θi, a.is, a.js, a.transposed))

_buttermulxa(x, θs, θi, is, js, transposed) = accummulxa(x, θs, θi, is, js, transposed)[end]

function _buttermulax(θs, θi, is, js, transposed, x) 
	o = deepcopy(x)
	for i in n:-1:1
		_mulax!(o, θs[θi[i]], is[i], js[i], x, transposed)
	end
	xs
end

function _∇buttermulax(Δ, o, θs, θi, is, js, transposed, x) 
	n = length(is)
	Δ = deepcopy(Δ);
	δθ = similar(θs);
	for i in 1:n
		_mulax!(x, θs[θi[i]], is[i], js[i], x, -transposed)
		δθ[θi[i]] = _∇mulax(Δ, θs[θi[i]], is[i], js[i], x, transposed)
		_mulax!(Δ, θs[θi[i]], is[i], js[i], Δ, -transposed)
	end
	(δθ, nothing, nothing, nothing, nothing, Δ)
end

function accummulxa(x, θs, θi, is, js, transposed) 
	n = length(is)
	xs = Vector{typeof(x)}(undef, n+1)
	xs[1] = deepcopy(x)
	for i in 1:n
		x = xs[i]
		o = deepcopy(x);
		_mulxa!(o, x, θs[θi[i]], is[i], js[i], transposed)
		xs[i+1] = o
	end
	xs
end

function _∇buttermulxa(Δ, xs, x, θs, θi, is, js, transposed) 
	n = length(is)
	Δ = deepcopy(Δ);
	δθ = similar(θs);
	for i in n:-1:1
		δθ[θi[i]] = _∇mulxa(Δ, xs[i], θs[θi[i]], is[i], js[i], transposed)
		 _mulxa!(Δ, Δ, θs[θi[i]], is[i], js[i], -transposed)
	end
	(δθ, Δ, nothing, nothing, nothing, nothing)
end


@adjoint function _buttermulax(θs, θi, is, js, transposed, x)
	xs = Unitary.accummulax(θs, θi, is, js, transposed, x)
	return xs[1], Δ -> _∇buttermulax(Δ, xs, θs, θi, is, js, transposed, x)
end

# @adjoint function _∇buttermulax(Δ, xs, θs, θi, is, js, transposed, x)
# 	return (_∇buttermulax(Δ, xs, θs, θi, is, js, transposed, x), Δ -> (nothing, nothing, nothing, nothing, nothing, nothing, nothing))
# end

@adjoint function _buttermulxa(x, θs, θi, is, js, transposed)
	xs = Unitary.accummulxa(x, θs, θi, is, js, transposed)
	return xs[end], Δ -> _∇buttermulxa(Δ, xs, x, θs, θi, is, js, transposed)
end



"""
	InPlaceUnitaryButterfly(n;indexes = :random)

	create a unitary matrix of dimension `n` parametrized by a set of givens rotations
	
	indexes --- method of generating indexes of givens rotations (`:butterfly` for the correct generation; `:random` for randomly generated patterns)
"""
function InPlaceUnitaryButterfly(a::UnitaryButterfly)
	n = a.matrices[1].n
	@assert all([m.n == n for m in a.matrices])
	θs = reduce(vcat, [m.θ for m in a.matrices])
	i = 1
	θi = map(a.matrices) do m 
		ii = (i:i + length(m.θ) - 1)
		i += length(m.θ)
		ii
	end
	is = [m.i for m in a.matrices]
	js = [m.j for m in a.matrices]
	InPlaceUnitaryButterfly(θs, θi, is, js, 1, n)
end

