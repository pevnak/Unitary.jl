using Unitary, Flux, BenchmarkTools
using Unitary: Butterfly, UnitaryButterfly

x = randn(Float32, 50, 100)
xx = randn(Float32, 100, 50)

a = Butterfly(50)
ta = transpose(a)
@btime a * x;		# 217.676 μs (4 allocations: 20.00 KiB)
@btime xx * a;		# 244.549 μs (4 allocations: 20.00 KiB)
@btime ta * x;		# 220.292 μs (4 allocations: 20.00 KiB
@btime xx * ta;		# 244.097 μs (4 allocations: 20.00 KiB)

ps = Flux.params(a)
@btime gradient(() -> sum(a * x), ps);	# 890.323 μs (58 allocations: 71.52 KiB)
@btime gradient(() -> sum(xx * a), ps);	# 473.158 μs (58 allocations: 71.52 KiB)
@btime gradient(() -> sum(ta * x), ps);	# 905.690 μs (66 allocations: 71.72 KiB)
@btime gradient(() -> sum(xx * ta), ps);# 476.453 μs (66 allocations: 71.72 KiB)


a = randn(Float32,50,50)
ta = transpose(a)
@btime a * x;		# 217.676 μs (4 allocations: 20.00 KiB)
@btime xx * a;		# 244.549 μs (4 allocations: 20.00 KiB)
@btime ta * x;		# 220.292 μs (4 allocations: 20.00 KiB
@btime xx * ta;		# 244.097 μs (4 allocations: 20.00 KiB)

ps = Flux.params(a)
@btime gradient(a -> sum(a * x), a);	# 190.613 μs (41 allocations: 70.64 KiB)
@btime gradient(a -> sum(xx * a), a);	# 189.654 μs (41 allocations: 70.64 KiB)
@btime gradient(a -> sum(transpose(a) * x), a);	# 190.524 μs (42 allocations: 70.66 KiB)
@btime gradient(a -> sum(xx * transpose(a)), a);# 186.843 μs (43 allocations: 70.67 KiB)

  
  
  
  
