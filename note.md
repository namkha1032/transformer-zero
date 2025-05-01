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

## xem xem co can phai train position embedding ko

## thu dung pytorch multiheadattention

## kiem tra xem neu chi dung 2 trong 3 mon query, key, value thi the nao

## mask chi ap dung cho decoder. Kiem tra lai xem khi test thi mask co ap dung ko

## kiem tra xem dong embedding co phai la learnable parameter ko

## xem xet reshape BTC to BCT https://grok.com/share/bGVnYWN5_a7023719-47c7-43bf-a58f-b9a91b9d9dbe --> phai chinh lai code trong generate

## chinh lai cipher, stoi, itos

## thu dung ModuleList de parallel query, key, value

## da co ffwd roi, sao lai can lm_head nua

## thu gom tat ca vao nn.Sequential

## xem lai clip giai thich ve residual connection (tam 1:28:00)

## Xem lai cai projection (Wo)

## Try code from scratch layer norm (1:35:00)







#########################################################

# query vector: what do I look for
# key vector: what do I contain
# --> affinity between tokens =  dot product between queries and keys
# query (B, T, 16)
# key (B, T, 16)
# weight aggregation now is data dependent
# before: wei is the same for all batches
# after: wei is different for each batch bc each batch has different tokens
# wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

# out = wei @ v
# Example: the 8th token knows what content it has, and what position it's in
# Base on that: the 8th token create a query: "I'm a vowel at 8th position, I'm looking for any consonant at positions up to four"
# All the nodes (tokens) emit the keys
# One of the channels could be "I'm a consonant at position up to 4" --> this key will have a high number in that specific channel
# ==> that's how the query and the key, when they dot product they can find each other and create high affinity
# through mask, we ensure that tokens do not look at the future
# through softmax, we can aggregate a lot of its information in to that position
# x is privaste information of each token
# - query: what do I look for
# - key: what do I have
# - value: if you find me interesting (high query*key), this is what I will communicate to you
# scaled attention is used to control the variance at initialization, so when we apply softmax, the output will stay diffuse and not lean towards highest number