import openai
import numpy as np
import operator
import copy
import math


def sentence_refiner(client,report,ground_truths):
    #help(openai.resources.embeddings.Embeddings)
    ground_truths = copy.deepcopy(ground_truths)
    ground_truths.append(report)
    resp = client.embeddings.create(
        input=ground_truths,
        model="text-embedding-ada-002")
    embedding_vector = resp.data
    report_embedding = embedding_vector.pop()
    # print(report_embedding.embedding)
    dots = list(map(lambda x:np.dot(report_embedding.embedding,x.embedding),embedding_vector))
    max_index, max_value = max(enumerate(dots), key=operator.itemgetter(1))
    return ground_truths[max_index]
