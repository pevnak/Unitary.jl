#wrapper for Zygote
function transposeldu(m)
	Matrix(transpose(m))
end

function transpose(a::Union{lowdup, inverted_lowdup})
	trans(a)
end

@adjoint function transpose(a::Union{lowdup, inverted_lowdup})
	Zygote.pullback(trans, a)
end

function trans(a::lowdup)
	m = transposeldu(a.m)
	lowdup(m)
end

function trans(a::inverted_lowdup)
	m = transposeldu(a.m)
	inverted_lowdup(m)
end

@adjoint function transposeldu(m)
	transposeldu(m), Δ -> (transpose(Δ), )
end
