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