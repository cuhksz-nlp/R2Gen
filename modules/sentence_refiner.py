import openai
import numpy as np
import operator


def sentence_refiner(report,ground_truths,key):
    client = openai.OpenAI(
        api_key=key
    )
    
    ground_truths = copy.deepcopy(ground_truths)
    #help(client.embeddings.create)
    original = (0 , None )
    shrink_data = 0.2
    threshold = 2
    max_iters = 10
    iters = 0
    prev_max = 0

    #intitialize
    zipped = list(zip(ground_truths,[0] * len(ground_truths)))

    while True:

        # get embedded vectors
        ground_truths = [i[0] for i in zipped ]

        resp = client.embeddings.create(
            input=(ground_truths+[report]),
            model="text-embedding-ada-002")

        # get embedded vectors dot distance
        embedding_vector = resp.data
        report_embedding = embedding_vector.pop()
        dots = list(map(lambda x:np.dot(report_embedding.embedding,x.embedding),embedding_vector))

        max_index, max_value = max(enumerate(dots), key=operator.itemgetter(1))

        if iters == 0 : original_value = max_value , ground_truths[max_index]

        if (len(dots) > 1 ) : print(f" Iters : {iters} , Values : {dots} , cur_Threshold: {abs(max_value - (sum(dots)/len(dots))) / np.std(dots)}")

        # check if fulfill target

        if len(dots) == 1 or abs(max_value - (sum(dots)/len(dots))) / np.std(dots) > threshold :

            if(max_value < original_value[0]):
                return original_value[1]
            else:
                return ground_truths[max_index]

        #check if exceed max iterations
        iters+=1
        if( iters >= max_iters ):
            print("exceed max iters , early terminated")
            if(max_value < original_value[0]):
                return original_value[1]
            else:
                return ground_truths[max_index]

        #no return , shrink data and recalculate
        zipped = list(zip(ground_truths,dots))
        to_shrink = math.ceil(len(zipped) * shrink_data)
        zipped.sort(key = lambda x:x[1])


        if to_shrink >= len(zipped):
            if(max_value < original_value[0]):
                return original_value[1]
            else:
                return ground_truths[max_index]

        zipped = zipped[to_shrink : ]
