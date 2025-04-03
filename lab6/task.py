import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# 1. Подготовка данных: создание графа
def create_graph():
    graph = {
        'A': ['B', 'C'],
        'B': ['C', 'F'],
        'C': ['A'],
        'D': ['A', 'B', 'E'],
        'E': ['D', 'F'],
        'F': ['E', 'C', 'B'],
    }
    nodes = list(graph.keys())
    return graph, nodes

# 2. Построение матрицы переходов
def build_transition_matrix(graph, nodes, alpha):
    N = len(nodes)
    M = np.zeros((N, N))
    
    for i, node in enumerate(nodes):
        outgoing = graph.get(node, [])
        num_outgoing = len(outgoing)
        if num_outgoing > 0:
            for target in outgoing:
                j = nodes.index(target)
                M[j, i] = 1.0 / num_outgoing
        else:
            M[:, i] = 1.0 / N
    
    M = alpha * M + (1 - alpha) / N * np.ones((N, N))
    return M

# 3. Реализация PageRank
def compute_pagerank(M, nodes, max_iter=100, tol=1e-6):
    N = len(nodes)
    pr = np.ones(N) / N
    history = [pr.copy()]
    
    for iteration in range(max_iter):
        pr_new = M.dot(pr)
        if np.max(np.abs(pr_new - pr)) < tol:
            print(f"Сошлось на итерации {iteration + 1}")
            break
        pr = pr_new
        history.append(pr.copy())
    
    return pr, history

# 4. Эксперименты с разными alpha
def run_experiments(graph, nodes):
    alphas = [0.3, 0.7, 0.85, 0.95]
    results = {}
    
    for alpha in alphas:
        print(f"\n=== Эксперимент с alpha = {alpha} ===")
        M = build_transition_matrix(graph, nodes, alpha)
        pr, history = compute_pagerank(M, nodes)
        results[alpha] = {'pr': pr, 'history': history}
        
        print("Итоговые PageRank:")
        for node, score in zip(nodes, pr):
            print(f"  {node}: {score:.4f}")
        print(f"Сумма PR: {np.sum(pr):.4f}")
    
    return results

# 5. Визуализация графа
def plot_graph(graph):
    G = nx.DiGraph()
    for node, targets in graph.items():
        for target in targets:
            G.add_edge(node, target)
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold', 
            arrows=True, arrowstyle='->', arrowsize=20)
    plt.title("Граф связей между страницами")
    plt.show()

# 6. Визуализация PageRank с субплотами
def plot_history(nodes, results):
    alphas = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, alpha in enumerate(alphas):
        history = np.array(results[alpha]['history'])
        ax = axes[idx]
        for i, node in enumerate(nodes):
            ax.plot(history[:, i], label=f"{node}")
        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Итерация")
        ax.set_ylabel("PageRank")
        ax.grid(True)
        ax.legend()
    
    plt.suptitle("Изменение PageRank по итерациям для разных α", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

graph, nodes = create_graph()
results = run_experiments(graph, nodes)
plot_history(nodes, results)

alpha = 0.85
print(f"\nТаблица изменений PageRank для α={alpha}:")
history = results[alpha]['history']
print("Итерация | " + " | ".join(nodes))
for i, pr in enumerate(history[:10]):
    print(f"{i:8d} | " + " | ".join(f"{x:.4f}" for x in pr))

plot_graph(graph)
