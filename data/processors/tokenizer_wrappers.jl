using PyCall

# Call py functions
pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
tokenizer_functions = pyimport("tokenizer_functions")

abstract type AbstractTokenizer end

# Fixed Tokenizers directly use transformers library from Hugging Face.
# As the name suggests, tokenizers are almost well-defined and preserved to change.
abstract type AbstractFixedTokenizer <: AbstractTokenizer end

# Flexible Tokenizers directly use tokenizers library from Hugging Face.
# As the name suggests, defining a tokenizer from scratch is welcomed and advised.
abstract type AbstractFlexibleTokenizer <: AbstractTokenizer end

# Fixed Tokenizers
# Fix 1: Bert Fixed
struct BertFixedTokenizer <: AbstractFixedTokenizer
    tokenizer::PyObject
end

BertFixedTokenizer(
    model_config::String = "bert-base-uncased",
    fast::Bool = false,
) = BertFixedTokenizer(
    tokenizer_functions.unsafe_setup_bert_fixed_tokenizer(model_config, fast),
)

# Flexible Tokenizers
# Flex 1: Bert Flexible
struct BertFlexibleTokenizer <: AbstractFlexibleTokenizer
    tokenizer::PyObject
    unk_token::String
    sep_token::String
    pad_token::String
    cls_token::String
    mask_token::String
end

# Either construct it from scratch...
BertFlexibleTokenizer(
    vocab_size::Int,
    unk_token = "[UNK]",
    sep_token = "[SEP]",
    pad_token = "[PAD]",
    cls_token = "[CLS]",
    mask_token = "[MASK]",
) = BertFlexibleTokenizer(
    tokenizer_functions.setup_bert_tokenizer(
        vocab_size,
        unk_token,
        sep_token,
        pad_token,
        cls_token,
        mask_token,
    ),
    unk_token,
    sep_token,
    pad_token,
    cls_token,
    mask_token,
)

# or load it from pretrained json
function load_flex_tokenizer(
    flex_type::Type{<:AbstractFlexibleTokenizer},
    tokenizer_path::String,
)
    tokenizer = tokenizer_functions.unsafe_load_tokenizer(tokenizer_path)
    return tokenizer
    flex_type(tokenizer, "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
end


function train_flex_tokenizer(flex::AbstractFlexibleTokenizer, trn_files)
    tokenizer_functions.train_tokenizer(flex.tokenizer, trn_files)
end

function save_flex_tokenizer(
    flex::AbstractFlexibleTokenizer,
    save_path::String = "/",
    tokenizer_name::String = string(typeof(flex)),
)
    tokenizer_functions.save_tokenizer(
        flex.tokenizer,
        save_path,
        tokenizer_name,
        flex.unk_token,
    )
end
