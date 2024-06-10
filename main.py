import numpy as np
import json
from matplotlib import pyplot as plt

atomic_numbers = []
atomicMasses = []
periods = []

with open('periodic-table.json', encoding='utf-8') as file:
    data = json.load(file)
    for element in data['elements']:
        atomic_numbers.append(element['number'])
        atomicMasses.append(element['atomic_mass'])
        periods.append(element['period'])

x_column = np.array(atomic_numbers).reshape(-1, 1)
ones_column = np.ones((x_column.shape[0], 1))
periods_column = np.array(periods).reshape(-1, 1)
y = np.array(atomicMasses)

def regression(X, y):
    # See explanation in README.md
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Linear Regression
X1 = np.hstack((x_column, ones_column))
b1 = regression(X1, y)
y1 = X1 @ b1
print(np.sqrt(np.mean((y - y1)**2)), np.round(b1, 2))

# Quadratic Regression
X2 = np.hstack((x_column**2, x_column, ones_column))
b2 = regression(X2, y)
y2 = X2 @ b2
print(np.sqrt(np.mean((y - y2)**2)), np.round(b2, 2))

# Multiple Regression
X3 = np.hstack((x_column, periods_column, ones_column))
b3 = regression(X3, y)
y3 = X3 @ b3
print(np.sqrt(np.mean((y - y3)**2)), np.round(b3, 2))

# Multiple Quadratic Regression
X4 = np.hstack((x_column**2, x_column, periods_column**2, periods_column, ones_column))
b4 = regression(X4, y)
y4 = X4 @ b4
print(np.sqrt(np.mean((y - y4)**2)), np.round(b4, 2))

# Visualize the data
plt.xlabel('Atomic number')
plt.ylabel('Atomic mass')
plt.title('Atomic mass vs Atomic number')
plt.scatter(atomic_numbers, atomicMasses)
plt.plot(atomic_numbers, y1, label='Linear')
plt.plot(atomic_numbers, y2, label='Quadratic')
plt.plot(atomic_numbers, y3, label='Multiple')
plt.plot(atomic_numbers, y4, label='Multiple Quadratic')
plt.legend()
# plt.show()
plt.savefig('results.png')