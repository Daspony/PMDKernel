using BenchmarkTools
using Random
include("../kernel.jl")
Random.seed!(1234)
const T = Float32

function setup_benchmark(n_points=1_000_000, n_dipoles=50_000)
    # Generate test data on host
    R_h = rand(T, n_points, 3) .* T(100)   # Evaluation points
    P_h = rand(T, 3, n_dipoles) .* T(50)   # Dipole positions
    M_h = rand(T, 3, n_dipoles)            # Dipole moments
    
    # Transfer to GPU (use column-major for better coalescing)
    R_d = CuArray(R_h)
    P_d = CuArray(P_h)
    M_d = CuArray(M_h)
    B_d = CUDA.zeros(T, 3, n_points)
    
    return R_d, P_d, M_d, B_d, n_points, n_dipoles
end

function benchmark_kernel(R_d, P_d, M_d, B_d, n, m, threads)
    # Configure launch
    blocks = cld(n, threads)
    shmem = 6 * BATCH_M * sizeof(T)  # 6 rows × BATCH_M
    
    println("=== Kernel Configuration ===")
    println("Threads/block: $threads")
    println("Blocks: $blocks")
    println("Shared memory/block: $(shmem ÷ 1024) KB")
    println("Total threads: $(threads * blocks)")
    println("Dipoles processed: $m (batch size: $BATCH_M)")
    
    # Warmup
    @cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(
        R_d, P_d, M_d, B_d, n, m
    )
    synchronize()
    
    # Timing with CUDA events (most accurate)
    println("\n=== Timing (CUDA Events) ===")
    start_event = CuEvent()
    end_event = CuEvent()
    
    CUDA.record(start_event)
    @cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(
        R_d, P_d, M_d, B_d, n, m
    )
    CUDA.record(end_event)
    synchronize(end_event)
    gpu_time = elapsed(start_event, end_event)
    
    println("Kernel time: $(gpu_time*1e3) ms")
    println("Throughput: $(n*m / gpu_time / 1e9) G interactions/sec")
    
    #= @benchmark for statistical robustness
    println("\n=== @benchmark Results ===")
    bench = @benchmark begin
        @cuda threads=$threads blocks=$blocks shmem=$shmem kernel_fused_B!(
            $R_d, $P_d, $M_d, $B_d, $n, $m
        )
        synchronize()
    end samples=10 evals=1
    
    display(bench)
    println("Median time: $(median(bench.times) / 1e6) ms")
    =#
    return #bench
end
