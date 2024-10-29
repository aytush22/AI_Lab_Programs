import networkx as nx

def create_semantic_network():
    G = nx.DiGraph()

    # Add nodes
    G.add_node("human")
    G.add_node("animal")
    G.add_node("bird")
    G.add_node("living_thing")
    G.add_node("man")
    G.add_node("woman")
    G.add_node("John")
    G.add_node("cat")
    G.add_node("fur")
    G.add_node("skin")
    G.add_node("move")
    G.add_node("giraffe")
    G.add_node("tall")
    G.add_node("long_legs")
    G.add_node("parrot")
    G.add_node("green")

    # Add edges
    G.add_edge("human", "living_thing", label="is_a")
    G.add_edge("animal", "living_thing", label="is_a")
    G.add_edge("bird", "animal", label="is_a")
    G.add_edge("living_thing", "breathes", label="can_do")
    G.add_edge("living_thing", "eats", label="can_do")
    G.add_edge("bird", "fly", label="can_do")
    G.add_edge("man", "human", label="is_a")
    G.add_edge("woman", "human", label="is_a")
    G.add_edge("John", "man", label="is_a")
    G.add_edge("cat", "animal", label="is_a")
    G.add_edge("cat", "fur", label="has")
    G.add_edge("animal", "skin", label="has")
    G.add_edge("animal", "move", label="can_do")
    G.add_edge("giraffe", "animal", label="is_a")
    G.add_edge("giraffe", "tall", label="is")
    G.add_edge("giraffe", "long_legs", label="has")
    G.add_edge("parrot", "bird", label="is_a")
    G.add_edge("parrot", "green", label="is")

    return G

# Create the semantic network
semantic_network = create_semantic_network()

# Visualize the network (optional)
import matplotlib.pyplot as plt
nx.draw(semantic_network, with_labels=True)
plt.show()