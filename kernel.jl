using CUDA
using LinearAlgebra

function kernel_fused_B!(R, P, M, B, n, m)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    # Register accumulators for the result
    Bx, By, Bz = 0.0f0, 0.0f0, 0.0f0
    
    # Definimos shared memory
    shmem = CuStaticSharedArray(Float32, (6, BATCH_M))
    
    Rx, Ry, Rz = 0.0f0, 0.0f0, 0.0f0
    if i <= n
        Rx = R[i, 1]
        Ry = R[i, 2]
        Rz = R[i, 3]
    end
    
    jb = 1
    while jb <= m
        batch_size = min(BATCH_M, m - jb + 1)
        
        # Cargamos la batch de dipolos al espacio reservado de shared memory
        total_elements = 6 * batch_size
        tid = threadIdx().x
        while tid <= total_elements

            col = (tid - 1) ÷ 6 + 1
            row = (tid - 1) % 6 + 1
            
            global_col = jb + col - 1
            
            if row <= 3
                shmem[row, col] = M[row, global_col]
            else
                shmem[row, col] = P[row - 3, global_col]
            end
            tid += blockDim().x 
        end
        
        # Aseguramos que todos los datos finalizaron de ser cargados a shmem antes de comenzar los calculos
        sync_threads()

        
        if i <= n
            for k in 1:batch_size

                μx, μy, μz = shmem[1, k], shmem[2, k], shmem[3, k]
                px, py, pz = shmem[4, k], shmem[5, k], shmem[6, k]
                
                dx = Rx - px
                dy = Ry - py
                dz = Rz - pz
                
                r2 = dx*dx + dy*dy + dz*dz
                r = sqrt(r2)
                
                # Protect against r=0
                if r > 1.0f-9 
                    inv_r3 = 1.0f0 / (r2 * r)
                    inv_r5 = inv_r3 / r2
                    dot_mr = dx*μx + dy*μy + dz*μz
    
                    scale = 1.0f-7 # μ0 / 4π
    
                    Bx += scale * (3.0f0 * dot_mr * dx * inv_r5 - μx * inv_r3)
                    By += scale * (3.0f0 * dot_mr * dy * inv_r5 - μy * inv_r3)
                    Bz += scale * (3.0f0 * dot_mr * dz * inv_r5 - μz * inv_r3)
                end
            end
        end
        
        # Aseguramos que todos los threads terminen antes de la siguiente batch
        sync_threads()
        jb += BATCH_M
    end

    if i <= n
        B[i, 1], B[i, 2], B[i, 3] = Bx, By, Bz
    end
    return
end