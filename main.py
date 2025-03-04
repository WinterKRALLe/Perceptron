from src.data_preparation import load_dataset, split_dataset, get_numpy_data
from src.perceptron import perceptron_train, perceptron_predict
from src.visualization import plot_learning_curve, plot_3d_dataset, plot_decision_plane, print_results

# 1. Načtení datasetu
df = load_dataset("data/perceptron_dataset.csv")

# 2. Rozdělení na trénovací a testovací množinu
train_df, test_df = split_dataset(df)
X_train, y_train, X_test, y_test = get_numpy_data(train_df, test_df)

# 3. Trénování perceptronu
w, errors = perceptron_train(X_train, y_train, max_epochs=10000, eta=0.01)
print("\nNaučené váhy w:")
print(w)

# Uložení grafu průběhu učení
plot_learning_curve(errors, save_path="results/learning_curve.png")

# 4. Vyhodnocení na testovací sadě
y_pred = perceptron_predict(X_test, w)
accuracy = (y_test == y_pred).mean() * 100
print(f"\nPřesnost na testovací sadě: {accuracy:.2f}%")

# Výpis výsledků ve zadaném formátu
print_results(y_test, y_pred)

# 5. Vizualizace 3D datasetu
plot_3d_dataset(df, elev=30, azim=20, save_path="results/graf_datasetu.png")

# Vizualizace 3D grafu s vykreslenou separační hyperplochou
plot_decision_plane(df, w, elev=30, azim=20, save_path="results/decision_plane.png")
