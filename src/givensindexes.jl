using Random
givenses_subblock(δ, offset = 0) = collect(1:δ) .+ offset, collect(1:δ) .+ δ .+ offset
function givenses_column(n, δ)
	k = div(n, 2*δ)
	s = [givenses_subblock(δ, 2*(i-1)*δ ) for i in 1:k]
	reduce(vcat, [x[1] for x in s]), reduce(vcat, [x[2] for x in s])
end

function givenses(n) 
	p = [(i,j) for i in 1:n for j in i+1:n]
	r = []
	while !isempty(p)
		s, p = nonoverlapping(p)
		push!(r, s)
	end
	tuple(r...)
end

function nonoverlapping(p)
	r = Vector{Tuple{Int,Int}}()
	u = Vector{Int}()
	ii = fill(false, length(p))
	for (i, (k,l)) in enumerate(p)
		if k ∉ u && l ∉ u 
			append!(u, [k,l])
			ii[i] = true
		end
	end
	return(tuple(p[ii]...), p[.!ii])
end
