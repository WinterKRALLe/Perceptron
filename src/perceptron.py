import numpy as np


def perceptron_train(X, y, max_epochs=10000, eta=0.01):
    """
    Trénuje perceptron na datech X, y (X včetně sloupce 1 pro bias).
    Vrací naučené váhy w a průběh počtu chyb v jednotlivých epochách.
    """
    w = np.random.randn(X.shape[1]) * 0.01
    errors_per_epoch = []
    n_samples = X.shape[0]
    
    for epoch in range(max_epochs):
        error_count = 0
        
        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]
            net = np.dot(w, x_i)
            y_pred = 1 if net >= 0 else 0
            
            if y_pred != y_i:
                error_count += 1
                w += eta * (y_i - y_pred) * x_i
        
        errors_per_epoch.append(error_count)
        if error_count == 0:
            print(f"\nTrénování ukončeno v {epoch+1}. epoše s nulovou chybou.")
            break
    
    return w, errors_per_epoch


def perceptron_predict(X, w):
    """Predikuje třídu (0 nebo 1) pro každý řádek X na základě vah w."""
    net = np.dot(X, w)
    return np.where(net >= 0, 1, 0)
