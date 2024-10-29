class predicate_KnowledgeBase:
    def __init__(self):
        self.facts = set()
        self.rules = []

    def add_fact(self, fact):
        self.facts.add(fact)

    def add_rule(self, premise, conclusion):
        self.rules.append({"premise": premise, "conclusion": conclusion})

    def is_fact(self, fact):
        return fact in self.facts

    def infer(self, query):
        for rule in self.rules:
            if rule['premise'] in self.facts and rule['conclusion'] not in self.facts:
                self.add_fact(rule['conclusion'])
                print(f"Inferred: {rule['conclusion']} based on {rule['premise']}")
        return self.is_fact(query)

# Example usage
if __name__ == "__main__":
    kb = predicate_KnowledgeBase()

    kb.add_fact("Alice is a parent of Bob")
    kb.add_fact("Bob is a parent of Charlie")

    # Define rules
    kb.add_rule("Alice is a parent of Bob", "Bob is a child of Alice")
    kb.add_rule("Bob is a parent of Charlie", "Charlie is a grandchild of Alice")

    query = "Charlie is a grandchild of Alice"
    if kb.infer(query):
        print(f"{query} is known.")
    else:
        print(f"{query} is not known.")
        
    print("Facts in KB:")
    for fact in kb.facts:
        print(f"- {fact}")
