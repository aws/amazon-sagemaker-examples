import logging
from itertools import groupby


def lambda_handler(event, context):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug("{}".format(event))

    text = event['dataObject']['source']

    token_values = PreProcessNERAnnotation().process(event)

    result = {
        "taskInput": {"tokens": token_values, "text": text},
        "humanAnnotationRequired": True
    }
    logger.debug("{}".format(result))
    return result


class PreProcessNERAnnotation:
    def __init__(self):
        pass

    def _split_and_get_index(self, text, sep=' '):
        p = 0
        for k, g in groupby(text, lambda x: x == sep):
            q = p + sum(1 for i in g)
            if not k:
                yield p, q
            p = q

    def process(self, event):
        text = event['dataObject']['source']
        # TODO: use nltk tokeniser..

        token_positions = self._split_and_get_index(text, ' ')

        token_values = [{"id": i, "startindex": p, "tokentext": text[p:q]} for i, (p, q) in enumerate(token_positions)]

        return token_values
