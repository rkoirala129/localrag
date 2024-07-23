# import faiss
# print(faiss.__version__)

# import faiss
# import numpy as np

# # Check if GPU version is available
# try:
#     res = faiss.StandardGpuResources()
#     print("FAISS GPU version is available")
    
#     # Create a small index as a test
#     d = 64                           # dimension
#     nb = 100000                      # database size
#     nq = 10000                       # nb of queries
#     np.random.seed(1234)             # make reproducible
#     xb = np.random.random((nb, d)).astype('float32')
#     xq = np.random.random((nq, d)).astype('float32')

#     index = faiss.IndexFlatL2(d)     # build the index
#     gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
#     gpu_index.add(xb)                # add vectors to the index
#     print(gpu_index.ntotal)

#     k = 4                            # we want to see 4 nearest neighbors
#     D, I = gpu_index.search(xq, k)   # actual search
#     print(I[:5])                     # neighbors of the 5 first queries
#     print(D[:5])                     # distances of the 5 first queries

# except AttributeError:
#     print("FAISS GPU version is not available")

# i

import faiss
import numpy as np

d = 64                           # dimension
nb = 100                      # database size
xb = np.random.random((nb, d)).astype('float32')

res = faiss.StandardGpuResources()  # use GPU
index = faiss.GpuIndexFlatL2(res, d)
index.add(xb)                    # add vectors to the index
print(index.ntotal)