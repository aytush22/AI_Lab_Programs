class Node:
    def __init__(self, value=None):
        self.value = value  
        self.children = [] 

    def add_child(self, child_node):
        """Add a child node to the current node."""
        self.children.append(child_node)

def alpha_beta(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or not node.children: 
        return node.value 

    if maximizing_player:
        max_eval = float('-inf')
        for child in node.children:
            eval = alpha_beta(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:  # Prune
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.children:
            eval = alpha_beta(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: 
                break
        return min_eval

# Example usage
if __name__ == "__main__":
    root = Node() 
    child1 = Node()  
    child2 = Node()  

    root.add_child(child1)
    root.add_child(child2)

    child1.add_child(Node(value=3))
    child1.add_child(Node(value=5))
    child1.add_child(Node(value=6))

    child2.add_child(Node(value=1))
    child2.add_child(Node(value=0))
    child2.add_child(Node(value=2))

    best_value = alpha_beta(root, depth=2, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
    print(f"The best value for the maximizing player is: {best_value}")
