import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for sequence_key, (X, lengths) in test_set.get_all_Xlengths().items():
        sequence_probabilities = {}
        for model_key, model in models.items():
            try:
                score = model.score(X, lengths)
                sequence_probabilities[model_key] = score
            except Exception as e:
                sequence_probabilities[model_key] = float("-inf")
                print(e)

        probabilities.append(sequence_probabilities)

    for values in probabilities:
        guesses.append(max(values, key=values.get))

    return probabilities, guesses
