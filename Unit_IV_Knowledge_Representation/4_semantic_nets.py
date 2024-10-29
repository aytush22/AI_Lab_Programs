#semantic nets
class Node:
    def __init__(self, name):
        self.name = name
        self.relationships = {}

    def add_relationship(self, relation, node):
        self.relationships[relation] = node

    def display(self, level=0):
        indent = '  ' * level
        print(f"{indent}{self.name}")
        for relation, node in self.relationships.items():
            print(f"{indent}  {relation} ->", end=" ")
            node.display(level + 1)

lion = Node("Lion")
elephant = Node("Elephant")
tiger = Node("Tiger")

carnivore = Node("Carnivore")
herbivore = Node("Herbivore")
savanna = Node("Savanna")
grassland = Node("Grassland")

lion.add_relationship("has_diet", carnivore)
lion.add_relationship("lives_in", savanna)
elephant.add_relationship("has_diet", herbivore)
elephant.add_relationship("lives_in", grassland)
tiger.add_relationship("has_diet", carnivore)
tiger.add_relationship("lives_in", savanna)

print("Semantic Network:")
lion.display()
elephant.display()
tiger.display()
