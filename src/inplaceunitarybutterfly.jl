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


function _buttermulax(θs, is, js, transposed, x) 
	o = deepcopy(x);
	for i in length(is):-1:1
		 _mulax!(o, θs[:,i], is[i], js[i], o, transposed)
	end
	o 
end

function _buttermulxa(x, θs, is, js, transposed) 
	o = deepcopy(x);
	for i in 1:length(is)
		 _mulxa!(o, o, θs[:,i], is[i], js[i], transposed)
	end
	o 
end

function _∇buttermulax(Δ, θs, is, js, transposed, x) 
	δx = deepcopy(Δ);
	δθ = similar(θs);
	for i in length(is):-1:1
		δθ[:,i] = _∇mulax!(o, θs[:,i], is[i], js[i], o, transposed)
		 _mulax!(o, θs[:,i], is[i], js[i], o, transposed)
	end
	o 
end


# @adjoint function _buttermulax(θs, is, js, x, t)
# 	return _buttermulax(θs, is, js, x, t) , Δ -> (_∇mulax(Δ, θs, is, js, x, t), nothing, nothing, _buttermulax(θs, is, js, Δ, -t))
# end

# @adjoint function _mulxa(x, θs, is, js, t)
# 	return _mulxa(x, θs, is, js, t) , Δ -> (_mulxa(Δ, θs, is, js, -t), _∇mulxa(Δ, x, θs, is, js, t), nothing, nothing, nothing)
# end


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

