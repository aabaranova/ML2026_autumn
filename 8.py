import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import oracles
import optimization

def run_on_dataset(path):
    print(f"\n--- Запуск на {path} ---")
    A, b = load_svmlight_file(path)
    A = A.tocsr()
    m, n = A.shape
    regcoef = 1.0 / m
    oracle = oracles.LogRegL2Oracle(A, b, regcoef)
    x0 = np.zeros(n)

    print(f"Размерность: m={m}, n={n}, nnz={A.nnz}")

    # Градиентный спуск 
    _, msg_gd, h_gd = optimization.gradient_descent(
        oracle, x0, trace=True)
    print(f"GD: {msg_gd}, итераций: {len(h_gd['func'])}")

    # Метод Ньютона 
    _, msg_n, h_n = optimization.newton(
        oracle, x0, trace=True)
    print(f"Newton: {msg_n}, итераций: {len(h_n['func'])}")

    # График 1: f(x_k) от времени
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    f_star = min(min(h_gd['func']), min(h_n['func']))
    
    axes[0].semilogy(h_gd['time'], np.array(h_gd['func']) - f_star + 1e-16,
                     linewidth=3, label=f'GD ({msg_gd})')
    axes[0].semilogy(h_n['time'], np.array(h_n['func']) - f_star + 1e-16,
                     linewidth=3, label=f'Newton ({msg_n})')
    axes[0].set_xlabel('Время, сек')
    axes[0].set_ylabel(r'$f(x_k) - f^*$')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].set_title('Значение функции от времени')

    # График 2: Относительный квадрат нормы градиента от времени
    g0 = h_gd['grad_norm'][0] ** 2
    axes[1].semilogy(h_gd['time'], np.array(h_gd['grad_norm'])**2 / g0,
                     linewidth=3, label=f'GD ({msg_gd})')
                     
    g0_n = h_n['grad_norm'][0] ** 2
    axes[1].semilogy(h_n['time'], np.array(h_n['grad_norm'])**2 / g0_n,
                     linewidth=3, label=f'Newton ({msg_n})')
                     
    axes[1].set_xlabel('Время, сек')
    axes[1].set_ylabel(r'$\|\nabla f(x_k)\|^2 / \|\nabla f(x_0)\|^2$')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].set_title('Норма градиента от времени')

    plt.suptitle(f'Датасет: {path}', fontsize=14)
    plt.tight_layout()
    plt.show()

# Запуск на датасетах
for name in ['w8a.txt', 'real-sim.bz2', 'gisette_scale.bz2']:
    run_on_dataset(f'data/{name}')
