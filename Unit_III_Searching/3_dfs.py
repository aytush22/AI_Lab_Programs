graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    print(node, end=" ") 

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

start_node = 'A'
print("DFS:")
dfs(graph, start_node)
