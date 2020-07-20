import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data    
path = 'C:\\Users\\goldenshop\\Desktop\\Project\\data.txt'
data = pd.read_csv(path, header=None, names=['Taille', 'Chambres', 'Prix'])

#afficher les données
print('data = ')
print(data.head(10) )
print()
print('data.describe = ')
print(data.describe())

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost




# redimensionnement des données '-1 <= x <= 1'
data = (data - data.mean()) / data.std()

print()
print('données après normalisation =')
print(data.head(10) )


# ajouter une colonne de '1'
data.insert(0, 'Ones', 1)


# séparer X (données d'entraînement) de y (variable cible)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


print('**************************************')
print('X data = \n' ,X.head(10) )
print('y data = \n' ,y.head(10) )
print('**************************************')


# convertir en matrices et initialiser thêta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))


print('X \n',X)
print('X.shape = ' , X.shape)
print('**************************************')
print('theta \n',theta)
print('theta.shape = ' , theta.shape)
print('**************************************')
print('y \n',y)
print('y.shape = ' , y.shape)
print('**************************************')


# initialiser les variables pour le taux d'apprentissage et les itérations
alpha = 0.1
iters = 100

# effectuer une régression linéaire sur l'ensemble de données
g, cost = gradientDescent(X, y, theta, alpha, iters)

# obtenir le coût (erreur) du modèle
thiscost = computeCost(X, y, g)


print('g = ' , g)
print('cost  = ' , cost[0:50] )
print('computeCost = ' , thiscost)
print('**************************************')


# obtenez la meilleure ligne d'ajustement pour la taille par rapport au prix

x = np.linspace(data.Taille.min(), data.Taille.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x)
print('f \n',f)

# tracer la ligne pour Taille vs Prix

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prédiction')
ax.scatter(data.Taille, data.Prix, label='Données d''entraînement')
ax.legend(loc=2)
ax.set_xlabel('Taille')
ax.set_ylabel('Prix')
ax.set_title('Taille vs. Prix')


# obtenez la meilleure ligne d'ajustement pour les chambres par rapport aux prix

x = np.linspace(data.Chambres.min(), data.Chambres.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x)
print('f \n',f)

# tracer la ligne pour Chambres vs Prix

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prédiction')
ax.scatter(data.Chambres, data.Prix, label='Données d''entraînement')
ax.legend(loc=2)
ax.set_xlabel('Chambres')
ax.set_ylabel('Prix')
ax.set_title('Taille vs. Prix')



# dessiner un graphique d'erreur

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Itérations')
ax.set_ylabel('Coût')
ax.set_title('Erreur par rapport à l''époque d''entraînement')