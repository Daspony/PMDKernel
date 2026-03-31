using NPZ
using Printf
using Distributions
using DelimitedFiles
using GPUArrays: @allowscalar
include("kernel.jl")
include("utils/gru.jl")
include("utils/bench.jl")

const BATCH_M =  64

function B0(viz::Bool, threads = 256)
    # Generación de la grilla de puntos R a evaluar
    gx = -60:1:60; gy = gx; gz = gx
    n = length(gx) * length(gy) * length(gz)

    Nx = length(gx)
    Ny = length(gy)
    Nz = length(gz)

    xmin = first(gx)
    ymin = first(gy)
    zmin = first(gz)

    R_cpu = zeros(Float32, n, 3)
    idx = 1
    for xi in gx, yi in gy, zi in gz
        R_cpu[idx, 1] = xi
        R_cpu[idx, 2] = yi
        R_cpu[idx, 3] = zi
        idx += 1
    end

    # Carga de datos y preparación de matrices de posición y momento
    data_path = joinpath(@__DIR__, "..", "data", "B0.npz")
    data = npzread(data_path)
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

        coordenadas_sensores = readdlm("coordenadas_sensores.csv", ",", Float32)

        open("Bsensores.csv", "a") do io
            for i in 1:size(coordenadas_sensores, 1)
                sensor = coordenadas_sensores[i, :]
                radio = sensor[1]
                theta = sensor[2]
                sensor[1] = (radio*sind(theta)) * 0.001f0
                sensor[2] = (radio*cosd(theta)) * 0.001f0
                sensor[3] = sensor[3] * 0.001f0

                xdecimal, xaprox = modf(sensor[1])
                ydecimal, yaprox = modf(sensor[2])
                zdecimal, zaprox = modf(sensor[3])

                if xdecimal >= 0.5; xaprox += 1; end
                if ydecimal >= 0.5; yaprox += 1; end
                if zdecimal >= 0.5; zaprox += 1; end

                xi_idx = Int(xaprox - xmin + 1)
                yi_idx = Int(yaprox - ymin + 1)
                zi_idx = Int(zaprox - zmin + 1)

                if xi_idx < 1 || xi_idx > Nx ||
                    yi_idx < 1 || yi_idx > Ny ||
                    zi_idx < 1 || zi_idx > Nz
                    continue
                end

                idx = (xi_idx - 1)*Ny*Nz + (yi_idx - 1)*Nz + zi_idx

                By_value = B_res[2, idx] * 1000

                writedlm(io, [By_value], ',')
            end
        end
    else
        benchmark_kernel(R, P, M, B, n, m, threads)
    end
end
