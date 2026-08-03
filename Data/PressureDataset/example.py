import sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression

'''
Example usage of packed bed hydraulics dataset.
'''

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
DATA_PATH = Path(__file__).with_name("pressure_raw.csv")

from Basic.newton_1d_fire_documentation import NM

def data_loader(csv_path=DATA_PATH, particle_diameter=5.0, diameter_tolerance=0.5):
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=2, usecols=(2, 3, 10, 13), dtype=float)

    column_length_mm = raw[:, 0]
    particles_diameter_mm = raw[:, 1]
    superficial_velocity = raw[:, 2]
    pressure_drop_pa = raw[:, 3]
    average_pressure_gradient = -1 * pressure_drop_pa / (column_length_mm / 1000.0) # flipping the sign for physical consistency

    filtered = np.column_stack((average_pressure_gradient, particles_diameter_mm, superficial_velocity))
    near_particle_diameter = np.isclose(particles_diameter_mm, particle_diameter, atol=diameter_tolerance)

    return filtered[near_particle_diameter]

def build_provider(X, y):
    X = X.reshape(-1,1)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    c1, c2, c3 = model.named_steps['linear'].coef_
    derivative = lambda x: c1 + 2*c2*x + 3*c3*x**2
    def evaluate(s, T=None, x=None):
        q = float(model.predict(np.array([[s]]))[0])
        dqds = float(derivative(s))
        return q, dqds, 0.0
    return evaluate

def packed_bed_example():
    '''
    Pressure gradient dp/dx in analogy with temperature gradient dT/dx; velocity corresponding to heat flux q. 
    Here flux/velocity is provided by surrogate (polynomial regression) trained on dataset.
    '''
    data = data_loader()
    provider = build_provider(data[:,0], data[:,2])
    source = lambda T,xg: 0.0
    x = np.linspace(0.0, 1.0, 21)
    U, log, num_iterations = NM(x, provider, source, T_dirichlet_left=3000.0, T_dirichlet_right=0.0, verbose=True)
    print(f"\nResult:\n{U}")
    print(f"\nNum iterations: {num_iterations}")


if __name__ == "__main__":
    packed_bed_example()
