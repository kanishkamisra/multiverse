from collections import defaultdict

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
    
    def get_leaves(self):
        desc = []
        if not self.children:
            yield self
        for child in self.children:
            for leaf in child.get_leaves():
                yield leaf
    
    def descendants(self):
        leaves = [leaf.value for leaf in self.get_leaves()]
        return leaves

class Nodeset(defaultdict):
    def __missing__(self, key):
        self[key] = new = self.default_factory(key)
        return new