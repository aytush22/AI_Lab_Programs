class Frame:
    def __init__(self, name, species, diet, habitat):
        self.name = name
        self.species = species
        self.diet = diet
        self.habitat = habitat

    def display(self):
        print(f"Name: {self.name}")
        print(f"Species: {self.species}")
        print(f"Diet: {self.diet}")
        print(f"Habitat: {self.habitat}")
        print()

class AnimalFrame(Frame):
    def __init__(self, name, species, diet, habitat, sound):
        super().__init__(name, species, diet, habitat)
        self.sound = sound

    def make_sound(self):
        print(f"The {self.name} says {self.sound}")

lion = AnimalFrame(name="Lion", species="Panthera leo", diet="Carnivore", habitat="Savanna", sound="Roar")
elephant = AnimalFrame(name="Elephant", species="Loxodonta africana", diet="Herbivore", habitat="Grassland", sound="Trumpet")

lion.display()
elephant.display()

lion.make_sound()
elephant.make_sound()
