def pad_collate(samples:BatchSamples, pad_idx:int=1, pad_first:bool=True) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding."
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(max_len, len(samples)).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[-len(s[0]):,i] = LongTensor(s[0])
        else:         res[:len(s[0]):,i] = LongTensor(s[0])
    return res, tensor([s[1] for s in samples])


