import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(errors, save_path="results/learning_curve.png"):
    """Vykreslí graf průběhu globální chyby v jednotlivých epochách."""
    plt.figure(figsize=(6,4))
    plt.plot(errors)
    plt.title("Průběh chyby perceptronu v jednotlivých epochách")
    plt.xlabel("Počet epoch")
    plt.ylabel("Počet chyb")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_3d_dataset(df, elev=30, azim=20, save_path="results/graf_datasetu.png"):
    """Vykreslí 3D graf datasetu podle sloupců Temperature, Light a CO2."""
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    df_occ0 = df[df['Occupancy'] == 0]
    df_occ1 = df[df['Occupancy'] == 1]
    
    ax.scatter(df_occ0['Temperature'], df_occ0['Light'], df_occ0['CO2'],
               color='red', label='Occupancy 0', alpha=0.6)
    ax.scatter(df_occ1['Temperature'], df_occ1['Light'], df_occ1['CO2'],
               color='green', label='Occupancy 1', alpha=0.6)
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Light')
    ax.set_zlabel('CO2')
    ax.set_title('3D graf datasetu')
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    
    plt.savefig(save_path)
    plt.close()


def plot_decision_plane(df, w, elev=30, azim=20, save_path="results/decision_plane.png"):
    """Vykreslí 3D graf datasetu a separační hyperplochu definovanou váhami perceptronu."""
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vykreslení dat
    df_occ0 = df[df['Occupancy'] == 0]
    df_occ1 = df[df['Occupancy'] == 1]
    
    ax.scatter(df_occ0['Temperature'], df_occ0['Light'], df_occ0['CO2'],
               color='red', label='Occupancy 0', alpha=0.6)
    ax.scatter(df_occ1['Temperature'], df_occ1['Light'], df_occ1['CO2'],
               color='green', label='Occupancy 1', alpha=0.6)
    
    # Vytvoření mřížky pro Temperature a Light
    temp_range = np.linspace(df['Temperature'].min(), df['Temperature'].max(), 10)
    light_range = np.linspace(df['Light'].min(), df['Light'].max(), 10)
    Temp_grid, Light_grid = np.meshgrid(temp_range, light_range)
    
    if abs(w[3]) > 1e-9:
        # Výpočet CO2 pro separační rovinu
        CO2_grid = -(w[0] + w[1]*Temp_grid + w[2]*Light_grid) / w[3]
        ax.plot_surface(Temp_grid, Light_grid, CO2_grid, alpha=0.3, color='brown')
    else:
        print("Varování: w[3] je (téměř) nulové, rovinu nelze vykreslit.")
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Light')
    ax.set_zlabel('CO2')
    ax.set_title('3D graf s hyperplochou perceptronu')
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    
    plt.savefig(save_path)
    plt.close()


def print_results(y_test, y_pred):
    """Vypíše výsledky testování ve zadaném formátu."""
    dataset_size = len(y_test)
    correct = (y_test == y_pred).sum()
    accuracy_percent = (correct / dataset_size) * 100
    wrong = dataset_size - correct
    wrong_percent = (wrong / dataset_size) * 100

    print(f"Dataset size | {dataset_size}")
    print("-" * 20)
    print(f"Correct      | {correct} | {accuracy_percent:.1f}%")
    print("-" * 20)
    print(f"Wrong        | {wrong} | {wrong_percent:.1f}%")
