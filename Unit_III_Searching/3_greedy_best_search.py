from queue import PriorityQueue
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 5), ('E', 12)],
    'C': [('F', 7)],
    'D': [],
    'E': [('G', 2)],
    'F': [('G', 1)],
    'G': []
}
heuristics = {
    'A': 10,
    'B': 4,
    'C': 6,
    'D': 7,
    'E': 2,
    'F': 3,
    'G': 0 
}

def greedy_best_first_search(start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((heuristics[start], start))
    
    print("Path:")
    while not pq.empty():
        _, current = pq.get()
        print(current, end=" ")
        if current == goal:
            print("\nGoal reached")
            return
        visited.add(current)
    
        for neighbor, _ in graph[current]:
            if neighbor not in visited:
                pq.put((heuristics[neighbor], neighbor))

start_node = 'A'
goal_node = 'G'
greedy_best_first_search(start_node, goal_node)