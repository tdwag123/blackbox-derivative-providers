from pathlib import Path
import numpy as np

'''
Prepare dataset for NM
'''

def main(particle_diameter=5.0, diameter_tolerance=0.5):
    csv_path = Path(__file__).with_name("pressure_raw.csv")
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=2, usecols=(2, 3, 10, 13), dtype=float)

    column_length_mm = raw[:, 0]
    particles_diameter_mm = raw[:, 1]
    superficial_velocity = raw[:, 2]
    pressure_drop_pa = raw[:, 3]
    average_pressure_gradient = -1 * pressure_drop_pa / (column_length_mm / 1000.0) # flipping the sign for physical consistency

    fill = np.ones(len(column_length_mm))
    filtered = np.column_stack((average_pressure_gradient, fill, fill, superficial_velocity))
    near_particle_diameter = np.isclose(particles_diameter_mm, particle_diameter, atol=diameter_tolerance)

    np.savetxt(
        f"pressure_filtered_{int(particle_diameter)}",
        filtered[near_particle_diameter],
        delimiter=",",
        header="s,T,q_clean,q_noisy",
        comments="",
    )

if __name__ == "__main__":
    # can pass in a different particle diameter/tolerance
    main()