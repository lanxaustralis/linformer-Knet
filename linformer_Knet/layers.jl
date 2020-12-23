using Knet
using LinearAlgebra: triu!
include("./operations.jl")
# TODO: tensor definition to stabilize fields
# TODO: must refer the linformer/attention all u need

array_type = Array{Float32}
Knet.param(d...; init = xavier_uniform, atype = array_type) = Param(atype(init(d...)))

# 1 - Embedding
struct Embed
    w::Any
end
Embed(vocabsize::Int, embedsize::Int) = Embed(param(embedsize, vocabsize))
(e::Embed)(x) = permutedims(e.w[:, x], (3, 1, 2)) .+ PE(size(x)[2], size(e.w)[1]) # returns tensor of the form T,E,B

function PE(inputsize::Int, embedsize::Int)
    PE = zeros(inputsize, embedsize)
    pos = collect(0:inputsize-1)

    for i = 1:Int(embedsize / 2)
        core = pos / 10^(5(2i - 2) / embedsize)
        PE[:, 2i-1] = sin.(core)
        PE[:, 2i] = cos.(core)
    end
    PE
end
# 2 - Dense
struct Dense
    w::Any
    b::Any
    σ::Function
end
Dense(inputsize::Int, outputsize::Int; σ = identity) =
    Dense(param(outputsize, inputsize), param(outputsize), σ)
(d::Dense)(x) = d.σ.(d.w * x .+ d.b)

# 3 - LayerNorm
struct LayerNorm
    γ::Any
    β::Any
end
LayerNorm(batchsize) = LayerNorm(param(batchsize), param(batchsize))
(ln::LayerNorm)(x) = one2three(ln.β, one2three(ln.γ, normalize(x, 1, 2)), f = +)

# FFN - refer Chain structure by 541
# TODO: ensure L1 & L2 regularization is needed
struct FFN
    L1::Dense
    L2::Dense
end
FFN(embed_dim::Int, ffn_depth::Int = 4) = FFN(
    Dense(embed_dim, embed_dim * ffn_depth, σ = relu),
    Dense(embed_dim * ffn_depth, embed_dim),
)
(ffn::FFN)(x) = reshape(
    permutedims(ffn.L2(ffn.L1(reshape(permutedims(x, (2, 1, 3)), size(x, 2), :))), (2, 1)),
    size(x),
)
# 4 - MHA
struct MHA
    projections::Array{Any,1}
end

# Naive MHA
MHA(embed_dim::Int; head_num::Int = 1, hidden_dim::Int = Int(floor(embed_dim / head_num))) =
    MHA(vcat(
        [param(embed_dim, hidden_dim, head_num) for i = 1:3],
        [param(head_num * hidden_dim, embed_dim)],
    ))

# MHLA
MHA(
    embed_dim::Int,
    proj_dim::Int,
    seq_length::Int;
    head_num::Int = 1,
    hidden_dim::Int = Int(floor(embed_dim / head_num)),
) = MHA(vcat(
    [param(embed_dim, hidden_dim, head_num) for i = 1:3],
    [param(head_num * hidden_dim, embed_dim)],
    [param(proj_dim, seq_length, head_num) for i = 1:2],
))

# TODO: Add arguments regarding different strategies of parameter sharing
function (mha::MHA)(
    cell;
    mem = (cell, cell),
    a_type = array_type,
    masked::Bool = false,
    mask_token::Float32 = -Inf32,
)
    linear = length(mha.projections) > 4

    T, E, B = size(cell)
    _, hidden, head_num = size(mha.projections[1])

    K, V = mem
    Q = cell

    head_container = a_type(undef, T, hidden, B, head_num)
    mask = a_type(undef, T, T - (linear ? (T - size(mha.projections[5], 1)) : 0))
    masked && (fill!(mask, mask_token); triu!(mask, 1))



    head(A, B, C) =
        masked ?
        bmm(
            softmax(
                bmm(A, permutedims(B, (2, 1, 3))) / sqrt(Float32(hidden)) .+ mask,
                dims = 2,
            ),
            C,
        ) :
        bmm(softmax(bmm(A, permutedims(B, (2, 1, 3))) / sqrt(Float32(hidden)), dims = 2), C)


    if linear
        for i = 1:head_num
            head_container[:, :, :, i] = head(
                three2two(Q, mha.projections[1][:, :, i]),
                three2two(
                    two2three(mha.projections[5][:, :, i], K),
                    mha.projections[2][:, :, i],
                ),
                three2two(two2three(mha.projections[6], V), mha.projections[3][:, :, i]),
            )
        end
    else
        for i = 1:head_num
            head_container[:, :, :, i] = head(
                three2two(Q, mha.projections[1][:, :, i]),
                three2two(K, mha.projections[2][:, :, i]),
                three2two(V, mha.projections[3][:, :, i]),
            )
        end
    end
    three2two(reshape(head_container, T, :, B), mha.projections[4])
end
