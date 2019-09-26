using Zygote: dropgrad
struct InPlaceUnitaryButterfly{T<:Real,I<:Tuple}
	θs::Matrix{T}
	is::Vector{I}
	js::Vector{I}
	transposed::Int
	n::Int
end

Flux.@treelike(InPlaceUnitaryButterfly)

Base.size(a::InPlaceUnitaryButterfly) = (a.n,a.n)
Base.size(a::InPlaceUnitaryButterfly, i::Int) = a.n

Base.eltype(a::InPlaceUnitaryButterfly{T,I}) where {T,I} = T
LinearAlgebra.transpose(a::InPlaceUnitaryButterfly) = InPlaceUnitaryButterfly(reverse(a.θs, dims = 2), reverse(a.is), reverse(a.js), -a.transposed, a.n)
Base.inv(a::InPlaceUnitaryButterfly) = transpose(a)
Base.show(io::IO, a::InPlaceUnitaryButterfly) = print(io, "$(a.n)x$(a.n) Unitary with $(size(a.θs,2)) butterfly matrices")
Base.zero(a::InPlaceUnitaryButterfly) = InPlaceUnitaryButterfly(zero(a.θ), a.i, a.j, a.transposed, a.n)

*(a::InPlaceUnitaryButterfly, x::TransposedMatVec) = (@assert size(x,1) == a.n; _buttermulax(a.θs, a.is, a.js, a.transposed, x))
*(x::TransposedMatVec, a::InPlaceUnitaryButterfly) = (@assert size(x,2) == a.n; _buttermulxa(x, a.θs, a.is, a.js, a.transposed))

_buttermulax(θs, is, js, transposed, x) = accummulax(θs, is, js, transposed, x)[1]
_buttermulxa(x, θs, is, js, transposed) = accummulxa(x, θs, is, js, transposed)[end]

function accummulax(θs, is, js, transposed, x) 
	n = length(is)
	xs = Vector{Matrix{eltype(x)}}(undef, n+1)
	xs[end] = deepcopy(x)
	for i in n:-1:1
		x = xs[i+1]
		o = deepcopy(x);
		_mulax!(o, θs[:,i], is[i], js[i], x, transposed)
		 xs[i] = o
	end
	xs
end

function accummulxa(x, θs, is, js, transposed) 
	n = length(is)
	xs = Vector{Matrix{eltype(x)}}(undef, n+1)
	xs[1] = deepcopy(x)
	for i in 1:n
		x = xs[i]
		o = deepcopy(x);
		_mulxa!(o, x, θs[:,i], is[i], js[i], transposed)
		xs[i+1] = o
	end
	xs
end

function _∇buttermulax(Δ, xs, θs, is, js, transposed, x) 
	n = length(is)
	Δ = deepcopy(Δ);
	δθ = similar(θs);
	for i in 1:n
		δθ[:,i] = _∇mulax(Δ, θs[:,i], is[i], js[i], xs[i+1], transposed)
		 _mulax!(Δ, θs[:,i], is[i], js[i], Δ, -transposed)
	end
	(δθ, nothing, nothing, nothing, Δ)
end

function _∇buttermulxa(Δ, xs, x, θs, is, js, transposed) 
	n = length(is)
	Δ = deepcopy(Δ);
	δθ = similar(θs);
	for i in n:-1:1
		δθ[:,i] = _∇mulxa(Δ, xs[i], θs[:,i], is[i], js[i], transposed)
		 _mulxa!(Δ, Δ, θs[:,i], is[i], js[i], -transposed)
	end
	(δθ, Δ, nothing, nothing, nothing)
end


@adjoint function _buttermulax(θs, is, js, transposed, x)
	xs = Unitary.accummulax(θs, is, js, transposed, x)
	return xs[1], Δ -> _∇buttermulax(Δ, xs, θs, is, js, transposed, x)
end

# @adjoint function _∇buttermulax(Δ, xs, θs, is, js, transposed, x)
# 	return (_∇buttermulax(Δ, xs, θs, is, js, transposed, x), Δ -> (nothing, nothing, nothing, nothing, nothing, nothing, nothing))
# end

@adjoint function _buttermulxa(x, θs, is, js, transposed)
	xs = Unitary.accummulxa(x, θs, is, js, transposed)
	return xs[end], Δ -> _∇buttermulxa(Δ, xs, x, θs, is, js, transposed)
end



"""
	InPlaceUnitaryButterfly(n;indexes = :random)

	create a unitary matrix of dimension `n` parametrized by a set of givens rotations
	
	indexes --- method of generating indexes of givens rotations (`:butterfly` for the correct generation; `:random` for randomly generated patterns)
"""
function InPlaceUnitaryButterfly(a::UnitaryButterfly)
	n = a.matrices[1].n
	@assert all([m.n == n for m in a.matrices])
	θs = reduce(hcat, [m.θ for m in a.matrices])
	is = [m.i for m in a.matrices]
	js = [m.j for m in a.matrices]
	InPlaceUnitaryButterfly(θs, is, js, 1, n)
end

