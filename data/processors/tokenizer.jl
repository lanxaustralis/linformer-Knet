# The Abstraction of the Vocabulary
struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer::Any
end

function Vocab(
    file::String;
    tokenizer = split,
    vocabsize = typemax(Int),
    mincount = 1,
    unk = "<unk>",
    eos = "<s>",
)
    vocab_freq = Dict{String,Int64}(unk => 1, eos => 1)
    w2i = Dict{String,Int64}(unk => 1, eos => 2) # Assign unk and eos tokens as 1 and 2, respectively
    i2w = Vector{String}()
    push!(i2w, unk, eos)

    open(file) do f
        for line in eachline(f)
            sentence = tokenizer(strip(lowercase(line)), [' '], keepempty = false)
            for word in sentence
                word == unk || word == eos && continue # They are default ones to be added later
                vocab_freq[word] = get!(vocab_freq, word, 0) + 1
            end
        end
        close(f)
    end


    vocab_freq = sort!(collect(vocab_freq), by = tuple -> last(tuple), rev = true)

    active_size = min(vocabsize - 2, length(vocab_freq)) # eos and unk ones excluded

    # trim to fit the size
    if length(vocab_freq) > active_size
        vocab_freq = vocab_freq[1:active_size]
    end

    # apply min count filter
    ind = active_size
    while ind > 0
        last(vocab_freq[ind]) >= mincount && break
        ind -= 1
    end
    vocab_freq = vocab_freq[1:ind]

    for i = 1:length(vocab_freq)
        ind = get!(w2i, first(vocab_freq[i]), 1 + length(w2i))
        length(i2w) < ind && push!(i2w, first(vocab_freq[i]))
    end

    Vocab(w2i, i2w, 1, 2, tokenizer)
end

# TODO: A smarter version of tokenizer could be implemented - or we can adapt an already exoisting strategy
