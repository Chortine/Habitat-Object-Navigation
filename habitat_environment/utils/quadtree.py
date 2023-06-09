# to fast detect a given xy is in area of which leaf. and make that leaf value True
# The value of a node represents that if this area has been visited

class Node:
    def __init__(self, val, is_leaf, t_left, t_right, b_left, b_right, coord, size):
        self.val = val
        self.is_leaf = is_leaf
        self.t_left = t_left
        self.t_right = t_right
        self.b_left = b_left
        self.b_right = b_right


# def construct_children_if_not_leaf(root):
#     root =


def main():
    # findout xy is at which node. if not, detect
    x = 0.34
    y = 0.87
    min_size = 1.0
    half_min_size = min_size / 2
    current_tree = Node(1, True, t_left=None, t_right=None, b_left=None, b_right=None,
                   coord=(x - half_min_size, y - half_min_size), size=min_size)


    x = 0.85
    y = 0.90
    # if not current_tree.check_in_tree_and_update(x, y):
    #     if
    #     new_tree = Node(False, )
