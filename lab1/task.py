import random
import time
import matplotlib.pyplot as plt

def linear_search(arr, key):
    for i in range(len(arr)):
        if arr[i] == key:
            return i
    return -1

def binary_search(arr, key):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def generate_sorted_array(size):
    return sorted([random.randint(1, 1000000) for _ in range(size)])

def measure_time(search_func, arr, key):
    start_time = time.perf_counter()  # Высокая точность
    search_func(arr, key)
    return time.perf_counter() - start_time

sizes = [10, 100, 1000, 10000, 100000]
results = []

for size in sizes:
    arr = generate_sorted_array(size)
    key = arr[len(arr) // 2]  # Ищем средний элемент

    # Замеряем время для каждого алгоритма
    linear_time = measure_time(linear_search, arr, key)
    binary_time = measure_time(binary_search, arr, key)

    results.append((size, linear_time, binary_time))

# Вывод таблицы
print("Размер массива | Линейный поиск (сек) | Двоичный поиск (сек)")
print("------------------------------------------------------------")
for size, lt, bt in results:
    print(f"{size:13} | {lt:.8f}           | {bt:.8f}")

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(
    [res[0] for res in results],
    [res[1] for res in results],
    marker='o',
    label='Линейный поиск'
)
plt.plot(
    [res[0] for res in results],
    [res[2] for res in results],
    marker='o',
    label='Двоичный поиск'
)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Размер массива (логарифмическая шкала)')
plt.ylabel('Время выполнения (сек, логарифмическая шкала)')
plt.title('Сравнение времени линейного и двоичного поиска')
plt.legend()
plt.grid(True)
plt.show()
