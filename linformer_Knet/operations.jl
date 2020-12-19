using Statistics: mean, std

# Normalization
function normalize(x; axis::Int = 1)
    μ = mean(x, dims = axis)
    σ = std(x, mean = μ, dims = axis)
    (x .- μ) ./ σ
end

isNormal(x) = isapprox(x, normalize(x))

# 3D tensor multiplied with 2D, resulting 3D
two2three(A, B) =
    (A == 1 ? B : A == 0 ? 0 : reshape(A * mat(B, dims = 1), (:, size(B)[2:end]...)))

# 3D tensor multiplied with 2D, resulting 3D
three2two(A, B) = (
    A == 1 ? B :
    A == 0 ? 0 : reshape(reshape(A, :, size(A, 2)) * B, (:, size(B, 2), size(A, 3)))
)
