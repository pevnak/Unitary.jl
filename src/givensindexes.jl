using Random
givenses_subblock(δ, offset = 0) = collect(1:δ) .+ offset, collect(1:δ) .+ δ .+ offset
function givenses_column(n, δ)
	k = div(n, 2*δ)
	s = [givenses_subblock(δ, 2*(i-1)*δ ) for i in 1:k]
	reduce(vcat, [x[1] for x in s]), reduce(vcat, [x[2] for x in s])
end

function givenses(n) 
	k = 2^(ceil(Int, log2(n)))
	δ = div(k, 2)
	filtergivenses.(n, _givenses(k, δ))
end

function _givenses(n, δ)
	δ < 1 && error("cannot return givens transformations smaller than one")
	δ == 1 && return((givenses_column(n,1),))
	(_givenses(n, div(δ,2))..., givenses_column(n, δ), _givenses(n, div(δ,2))...)
end

function filtergivenses(n, g::Tuple{Vector,Vector})
	i, j = g
	mask = (i .<= n) .& (j .<= n)
	i[mask], j[mask]
end


function randomgivenses(n)
	k = div(n, 2)
	idxs = map(1:n-1) do _ 
		p = randperm(n)
		(p[1:k], p[k+1:2k])
	end 
	tuple(idxs...)
end
