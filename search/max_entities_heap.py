import heapq


class MaxEntitiesHeap:
    """
    Max heap
    """

    def __init__(self, capacity, entities):
        self.heap = []
        self.capacity = capacity
        self.entities = entities

    def add(self, score, term):
        # if this term already exists in the query, skip it
        if self.is_term_exists_in_entities(term):
            return
        # if this term already exists in the heap, compare their similarities
        existing_items = [(i, v) for i, v in enumerate(self.heap) if v[1] in term]
        if existing_items:
            existing_item = existing_items[0]
            if score is not None and score > existing_item[1][0]:  # we're better -> let's replace the existing term
                self.heap[existing_item[0]] = existing_item[1]
                self.heapify()  # need to re-heapify
            else:  # the existing term is better
                return
        item = (score, term)
        if len(self.heap) >= self.capacity:
            heapq.heappushpop(self.heap, item)
        else:
            heapq.heappush(self.heap, item)

    def heapify(self):
        heapq.heapify(self.heap)

    def is_term_exists_in_entities(self, term):
        """
        Returns true iff a given terms exists in a given sequence of entities
        """
        term = term.replace('_nnp', '')
        for entity in self.entities:
            entity = entity.replace('_nnp', '')
            if term in entity:
                return True
        return False
