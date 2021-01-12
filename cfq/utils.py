from torch.nn.utils.rnn import PackedSequence


pack_as = lambda data, packed_seq: PackedSequence(data, packed_seq.batch_sizes, packed_seq.sorted_indices, packed_seq.unsorted_indices)
