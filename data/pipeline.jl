include("./processors/iterator.jl")

const dir = "./data/datasets/ptb-test"

example_file = string(dir, "/train.txt")
example_vocab = Vocab(example_file)
example_raw = TextReader(example_file,example_vocab)
example_processed_iter = NaiveLMData(example_raw)

example_final = collect(example_processed_iter)
