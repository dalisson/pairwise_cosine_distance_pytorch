import torch

def pairwise_similarity(matrix_of_vectors):
    '''
    Computes cosine similarities for between all vectors, extremely useful for comparing
    similarities between embeddings when doing deep embedding learning.

    input:
        matrix_of_vectors: tensor with shape (n_vectors, vector_size)

    output:
        similarities : tensor with shape (n_vector, n_vectors)
    Each row[i, j] is the similarity of the ith element against the jth vector, eg,
    row[0,0] is 1 and row[0,42] is the similarity between the first
    element in the input and the 43th element in the input.
    '''

    dot_product = matrix_of_vectors@matrix_of_vectors.t()
    norms = torch.sqrt(torch.einsum('ii->i', dot_product))
    similarities = dot_product/(norms[None]*norms[..., None])

    return similarities
    