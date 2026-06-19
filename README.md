this model uses what i call kmoment attention combined with standard self attention, includes engrammlp (explained in a different repo), uses mixture of experts, and 
uses what i call  a step pyramid attention. you only need a small embedding vector in early layers for semantics(update could be to not have long context in first few layers). Starts with full attention 
on local window and top_k for longer history, then deeper in layers do  top_k for both global and local context, increase embedding dimension for these later layers for richer computation on what is important.

I am sure someone else has already made the kmoment attention but it creates moments from the embedding vectors, acting like simplicial complexes on the embedding vectors,
and  does attention on those moments as well as on the normal embedding vectors. 

the results...
km_superstep_pyr_moe.py
vocab size  1000
size of model 12497996 bsize=64 lbsize = 256
step 0: train loss 7.0364, val loss 7.0356, lr 1.50e-06
step 5000: train loss 2.8515, val loss 3.2058, lr 2.59e-04
step 10000: train loss 2.5595, val loss 3.0742, lr 1.52e-04
step 15000: train loss 2.4236, val loss 3.0475, lr 4.48e-05
step 20000: train loss 2.3962, val loss 2.9756, lr 1.89e-12
Training time: 3246.07 seconds
