from collections import defaultdict
from .list_utils import intersect

class Node:
    def __init__(self, value: str):
        self.value = value
        self.children = []
        self.parent = None
    def __repr__(self):
        return f'Node {self.value}\nParent:{self.parent.value if self.parent is not None else None}\nChildren: {[c.value for c in self.children]}'
    
    def add_child(self, value):
        if value not in self.children:
            self.children.append(value)
            value.add_parent(self)
    
    def add_parent(self, value):
        if self.parent is None:
            self.parent = value
    
    def path(self):
        toroot = []
        node = self
        node_val = self.value
        toroot.append(node_val)
        while node.parent != None:
            toroot.append(node.parent.value)
            node = node.parent
        return toroot

    def ancestors(self):
        if self.parent is None:
            return[]
        else:
            path = self.path()[1:]
            return path
    
    def get_leaves(self):
        if not self.children:
            yield self
        for child in self.children:
            for leaf in child.get_leaves():
                yield leaf
    
    def descendants(self):
        leaves = [leaf.value for leaf in self.get_leaves()]
        return leaves

    def siblings(self, return_self = False):
        if self.parent is None:
            return []
        else:
            sibling_nodes = self.parent.children
            if return_self:
                siblings = [node.value for node in sibling_nodes]
            else:
                siblings = [node.value for node in sibling_nodes if node.value != self.value]
            return siblings

    def depth(self):
        path = self.path()
        return len(path)

    def height(self):
        if self == None:
            return 0
        elif self.children == []:
            return 1
        else:
            height = 0
            for child in self.children:
                height = max(height, child.depth())

        return height+1

class Nodeset(defaultdict):
    def __missing__(self, key):
        self[key] = new = self.default_factory(key)
        return new
    
    def get_root(self):
        try:
            assert self != {}
        except AssertionError:
            raise AssertionError("Nodeset currently empty!")
        return list(self.values())[0]

    def lcs(self, nodes):
        return intersect([self[node].path() for node in nodes])[0]

    def wup_sim(self, node1, node2):
        '''
        Generic Wu-Palmer Similarity
        '''
        lcs = self.lcs([node1, node2])
        lcs_depth = self[lcs].depth()

        depth1 = self[node1].depth()
        depth2 = self[node2].depth()
        avg_depth = (depth1 + depth2)/2

        return lcs_depth/(avg_depth)
    
    def generalized_wup_sim(self, nodes):
        '''
        Generalized Wu-Palmer Similarity
        '''
        lcs = self.lcs(nodes)
        lcs_depth = self[lcs].depth()

        depths = [self[node].depth() for node in nodes]
        avg_depth = sum(depths)/len(depths)

        return lcs_depth/(avg_depth)

    def path_len_sim(self, node1, node2):
        '''
        Path-len similarity between two nodes
        '''
        raise NotImplementedError