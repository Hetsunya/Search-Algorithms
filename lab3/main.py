import json
import heapq
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def heuristic_euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def load_maze(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def a_star_search(maze_data, heuristic, allow_diagonal=False):
    width, height = maze_data['width'], maze_data['height']
    start, goal = tuple(maze_data['start']), tuple(maze_data['goal'])
    maze = maze_data['maze']
    
    open_list = []
    heapq.heappush(open_list, (0, 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited_nodes = 0
    
    tiebreaker = 0
    while open_list:
        _, _, current = heapq.heappop(open_list)
        visited_nodes += 1
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, visited_nodes
        
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        if allow_diagonal:
            neighbors.extend([(x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)])
        
        for neighbor in neighbors:
            nx, ny = neighbor
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0:
                cost = 1.414 if allow_diagonal and (abs(nx - x) + abs(ny - y)) == 2 else 1
                tentative_g_score = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    tiebreaker += 1
                    heapq.heappush(open_list, (f_score[neighbor], tiebreaker, neighbor))
    
    return None, visited_nodes

def visualize_maze(maze_data, results, file_path):
    maze = np.array(maze_data['maze'])
    heuristics = list(results.keys())
    num_heuristics = len(heuristics)
    
    fig, axes = plt.subplots(1, num_heuristics, figsize=(num_heuristics * 5, 5))
    fig.suptitle(f"Лабиринт: {file_path.split('/')[-1]}", fontsize=14)
    
    if num_heuristics == 1:
        axes = [axes]
    
    for ax, heuristic_name in zip(axes, heuristics):
        path = results[heuristic_name]["path"]
        
        ax.imshow(maze, cmap='Greys', origin='upper')
        ax.set_xticks(np.arange(-0.5, maze_data['width'], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze_data['height'], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='both', size=0)

        ax.add_patch(Rectangle((maze_data['start'][0] - 0.5, maze_data['start'][1] - 0.5), 1, 1, color='green', alpha=0.8))
        ax.add_patch(Rectangle((maze_data['goal'][0] - 0.5, maze_data['goal'][1] - 0.5), 1, 1, color='red', alpha=0.8))

        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Путь')
            for x, y in path:
                ax.plot(x, y, 'bo', markersize=5)
        
        ax.set_title(f"{heuristic_name}\nПосещено: {results[heuristic_name]['visited_nodes']}\nДлина пути: {results[heuristic_name]['path_length']}\nВремя: {results[heuristic_name]['time']:.4f}с", fontsize=10)
        ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def compare_heuristics(file_paths):
    heuristics = {
        "Manhattan": (heuristic_manhattan, False),
        "Euclidean с диагональным движением": (heuristic_euclidean, True),
        "Euclidean без диагонального движения": (heuristic_euclidean, False)
    }
    all_results = {}

    for file_path in file_paths:
        maze_data = load_maze(file_path)
        results = {}
        
        for name, (heuristic, allow_diagonal) in heuristics.items():
            start_time = time.perf_counter()
            path, visited_nodes = a_star_search(maze_data, heuristic, allow_diagonal)
            elapsed_time = time.perf_counter() - start_time
            
            results[name] = {
                "path_length": len(path) if path else 0,
                "visited_nodes": visited_nodes,
                "time": elapsed_time,
                "path": path
            }
        
        all_results[file_path] = results
        visualize_maze(maze_data, results, file_path)
    
    print("\n=== Результаты сравнения эвристик ===")
    for file_path in file_paths:
        print(f"\nЛабиринт: {file_path.split('/')[-1]}")
        print(f"{'Эвристика':<30} | {'Длина пути':<12} | {'Посещено узлов':<15} | {'Время (с)':<10}")
        print("-" * 70)
        for name in heuristics:
            r = all_results[file_path][name]
            print(f"{name:<30} | {r['path_length']:<12} | {r['visited_nodes']:<15} | {r['time']:.6f}")
    
    return all_results

if __name__ == "__main__":
    file_paths = [
        "maze_10x10.json",
        "maze_20x20.json",
        "maze_nxm.json"
    ]
    results = compare_heuristics(file_paths)