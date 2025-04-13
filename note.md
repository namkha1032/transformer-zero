# preprocess
## trade-off between vocabulary size and encoded sequence length
- If we encode each character -> vocab size is around 65 -> encoded value of "hello world" has size 11 (number of characters)
- If we encode each word (token) -> vocab size is around 50.000 -> encoded value of "hello world" has size 3 (number of words)

## block_size = batch_size ? ==> not really
-> Make the transformer to be able to predict character from 1-character context to 8-character context (block size) --> after block size -> truncating because transformer never receive more than block-size input when predicting next character (around 16:00 in clip)

- when input is tensor([18]) the target: 47
- when input is tensor([18, 47]) the target: 56
- when input is tensor([18, 47, 56]) the target: 57
- when input is tensor([18, 47, 56, 57]) the target: 58
- when input is tensor([18, 47, 56, 57, 58]) the target: 1
- when input is tensor([18, 47, 56, 57, 58,  1]) the target: 15
- when input is tensor([18, 47, 56, 57, 58,  1, 15]) the target: 47
- when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target: 58

## Batch size
- `batch_size` = 4 # how many independent sequences will we process in parallel?
- `block_size` = 8 # what is the maximum context length for predictions?
- `xb`: the context
- `yb`: the predicted token (character)

## torch.multinomial

## xem lai xem tai sao phai dung layer norm