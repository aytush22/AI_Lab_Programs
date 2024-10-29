from collections import deque

class WaterJugProblem:
    def __init__(self, capacity_jug1, capacity_jug2, target):
        self.capacity_jug1 = capacity_jug1
        self.capacity_jug2 = capacity_jug2
        self.target = target

    def solve(self):
        queue = deque()
        visited = set()

        initial_state = (0, 0)
        queue.append(initial_state)
        visited.add(initial_state)

        while queue:
            current_state = queue.popleft()
            jug1, jug2 = current_state

            if jug1 == self.target or jug2 == self.target:
                return current_state 

            possible_states = [
                (self.capacity_jug1, jug2), 
                (jug1, self.capacity_jug2),  
                (0, jug2),
                (jug1, 0), 
                (max(0, jug1 - (self.capacity_jug2 - jug2)), min(self.capacity_jug2, jug1 + jug2)),
                (min(self.capacity_jug1, jug1 + jug2), max(0, jug2 - (self.capacity_jug1 - jug1))) 
            ]

            for state in possible_states:
                if state not in visited:
                    visited.add(state) 
                    queue.append(state) 

        return None 

# Example usage
if __name__ == "__main__":
    capacity_jug1 = 4 
    capacity_jug2 = 3 
    target = 2  

    problem = WaterJugProblem(capacity_jug1, capacity_jug2, target)
    result = problem.solve()

    if result:
        print(f"Reached target with jug states: {result}")
    else:
        print("No solution found.")
