using NPZ
using Printf
using Distributions
using GPUArrays: @allowscalar
include("kernel.jl")
include("utils/gru.jl")
include("utils/bench.jl")

const BATCH_M =  64

function B0(viz::Bool, threads = 256)
    # Generación de la grilla de puntos R a evaluar
    gx = -60:1:60; gy = gx; gz = gx
    n = length(gx) * length(gy) * length(gz)
    R_cpu = zeros(Float32, n, 3)
    idx = 1
    for xi in gx, yi in gy, zi in gz
        R_cpu[idx, 1] = xi
        R_cpu[idx, 2] = yi
        R_cpu[idx, 3] = zi
        idx += 1
    end

    # Carga de datos y preparación de matrices de posición y momento
    data = npzread("data/B0.npz")
    M1_cpu = Float32.(hcat(data["array1"], data["array3"]))
    M2_cpu = Float32.(hcat(data["array2"] .* 2.035f0, data["array4"] .* 3.051f0))

    n = size(R_cpu, 1)
    m = size(M2_cpu, 2)

    B_cpu = zeros(Float32, n, 3)

    R = CuArray(R_cpu .* 0.001f0)
    M = CuArray(M2_cpu)
    P = CuArray(M1_cpu .* 0.001f0)
    B = CuArray(B_cpu)

    blocks = cld(n, threads)
    shmem = 6 * BATCH_M * sizeof(Float32)

    if viz
        @cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(R, P, M, B, n, m)
        B_res = Array(B')

        XX = [xi for xi in gx, yi in gy, zi in gz]
        mask = trues(size(XX))
        By = zeros(size(XX))
        @allowscalar begin
            By[mask] = B_res[2,:] .* -1000 # mT, el menos es unicamente para invertir los colores del heatmap
            fig = Figure(size=(600,600))
            saxi = Slicer3D(fig,By,zoom=3)  
            display(fig)
        end
    else
        benchmark_kernel(R, P, M, B, n, m, threads)
    end
end