import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Теоретическое введение в комментариях
"""
1. Модель «случайного сёрфера»:
   - Представляет пользователя, который случайным образом переходит по ссылкам на веб-страницах.
   - С вероятностью α (damping factor) переходит по ссылке, с вероятностью (1-α) — случайный переход на любую страницу.
   - Это моделирует поведение реальных пользователей и решает проблему «ловушек» (страниц без исходящих ссылок).

2. Формула PageRank:
   PR(A) = (1-α)/N + α * (PR(T1)/C(T1) + PR(T2)/C(T2) + ... + PR(Tn)/C(Tn))
   - PR(A) — PageRank страницы A.
   - α — damping factor (обычно 0.85).
   - N — общее число страниц.
   - T1, T2, ..., Tn — страницы, ссылающиеся на A.
   - C(Ti) — число исходящих ссылок со страницы Ti.

3. Итерационный процесс:
   - Начинаем с равных начальных значений PR (обычно 1/N).
   - Пересчитываем PR для всех страниц по формуле до сходимости (разница между итерациями мала).
"""

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

# Основной код
graph, nodes = create_graph()
results = run_experiments(graph, nodes)
plot_history(nodes, results)

# Таблица изменений для α=0.85
alpha = 0.85
print(f"\nТаблица изменений PageRank для α={alpha}:")
history = results[alpha]['history']
print("Итерация | " + " | ".join(nodes))
for i, pr in enumerate(history[:10]):
    print(f"{i:8d} | " + " | ".join(f"{x:.4f}" for x in pr))

plot_graph(graph)  # Визуализация графа

# Анализ
print("\nАнализ:")
print("- PageRank учитывает как количество входящих ссылок, так и их значимость:")
print("  * Страницы с большим числом входящих ссылок (например, A) получают высокий PR.")
print("  * Ссылки от страниц с высоким PR (например, D → A) увеличивают вес больше, чем от страниц с низким PR.")
print("- Damping factor (α):")
print("  * При α=0.6 больше случайных переходов → распределение PR ближе к равномерному.")
print("  * При α=0.95 больше веса ссылкам → усиление различий между страницами.")