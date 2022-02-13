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
        if existing_items := [
            (i, v) for i, v in enumerate(self.heap) if v[1] in term
        ]:
            existing_item = existing_items[0]
            if score is None or score <= existing_item[1][0]:
                return
            self.heap[existing_item[0]] = existing_item[1]
            self.heapify()  # need to re-heapify
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
