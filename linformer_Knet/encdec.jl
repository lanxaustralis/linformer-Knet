using Knet
include("./layers.jl")

struct EncoderLayer
    mha::MHA
    ln1::LayerNorm
    ffn::FFN
    ln2::LayerNorm
end

EncoderLayer(
    embed_dim::Int;
    batch_size::Int = 1,
    head_num::Int = 1,
    hidden_dim::Int = Int(floor(embed_dim / head_num)),
    ffn_depth::Int = 4,
) = EncoderLayer(
    MHA(embed_dim; head_num = head_num, hidden_dim = hidden_dim),
    LayerNorm(batch_size),
    FFN(embed_dim, ffn_depth),
    LayerNorm(batch_size),
)

EncoderLayer(
    embed_dim::Int,
    seq_length::Int,
    proj_dim::Int;
    batch_size::Int = 1,
    head_num::Int = 1,
    hidden_dim::Int = Int(floor(embed_dim / head_num)),
    ffn_depth::Int = 4,
) = EncoderLayer(
    MHA(embed_dim, proj_dim, seq_length, head_num = head_num, hidden_dim = hidden_dim),
    LayerNorm(batch_size),
    FFN(embed_dim, ffn_depth),
    LayerNorm(batch_size),
)

function (enc::EncoderLayer)(
    input;
    # param_sharing::Bool = false,
    a_type = Array{Float32},
)
    input .+= enc.mha(input, a_type = a_type)
    input = enc.ln1(input)
    input .+= enc.ffn(input)
    enc.ln2(input)
end

struct DecoderLayer
    masked_mha::MHA
    ln1::LayerNorm
    mha::MHA
    ln2::LayerNorm
    ffn::FFN
    ln3::LayerNorm
end

DecoderLayer(
    embed_dim::Int;
    batch_size::Int = 1,
    head_num::Int = 1,
    hidden_dim::Int = Int(floor(embed_dim / head_num)),
    ffn_depth::Int = 4,
) = DecoderLayer(
    MHA(embed_dim, head_num = head_num, hidden_dim = hidden_dim),
    LayerNorm(batch_size),
    MHA(embed_dim, head_num = head_num, hidden_dim = hidden_dim),
    LayerNorm(batch_size),
    FFN(embed_dim, ffn_depth),
    LayerNorm(batch_size),
)

DecoderLayer(
    embed_dim::Int,
    seq_length::Int,
    proj_dim::Int;
    batch_size::Int = 1,
    head_num::Int = 1,
    hidden_dim::Int = Int(floor(embed_dim / head_num)),
    ffn_depth::Int = 4,
) = DecoderLayer(
    MHA(embed_dim, head_num = head_num, hidden_dim = hidden_dim),
    LayerNorm(batch_size),
    MHA(embed_dim, proj_dim, seq_length, head_num = head_num, hidden_dim = hidden_dim),
    LayerNorm(batch_size),
    FFN(embed_dim, ffn_depth),
    LayerNorm(batch_size),
)

function (dec::DecoderLayer)(
    input;
    # param_sharing::Bool = false,
    a_type = Array{Float32},
)
    input .+= dec.masked_mha(input, a_type = a_type, masked = true)
    input = dec.ln1(input)
    input .+= dec.mha(input, a_type = a_type)
    input = dec.ln2(input)
    input .+= dec.ffn(input)
    dec.ln3(input)
end


struct CoderStack{T}
    layers::Array{T,1}
    CoderStack(T, N::Int, args...; kwargs...) = new{T}([T(args...; kwargs...) for i = 1:N])
end

function (cs::CoderStack)(
    input;
    # param_sharing::Bool = false,
    a_type = Array{Float32},
)

    for i = 1:length(cs.layers)
        input = cs.layers[i](input)
    end
    input
end


# Comment out to test on place
# TODO: Must check whether mask,linear and other strategies works properly

# T, E, B, k = 20, 50, 30, 4
# input = param(T, E, B)
# # enc_lyr = EncoderLayer(E, B)
# # res = enc_lyr(input)
# enc_stack = CoderStack(EncoderLayer, 6, E, batch_size = B)
# res = enc_stack(input)
#
# lenc_stack = CoderStack(EncoderLayer, 6, E, T, k, batch_size = B)
# lres = lenc_stack(input)
#
# ## Dec stack
# dec_stack = CoderStack(DecoderLayer, 6, E, batch_size = B)
# decres = dec_stack(input)
#
# lenc_stack = CoderStack(DecoderLayer, 6, E, T, k, batch_size = B)
# declres = lenc_stack(input)
