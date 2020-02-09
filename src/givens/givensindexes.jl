using Random

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
