import matplotlib.pyplot as plt
import numpy as np


def solve_potential_2d_medium(
    grid_size=100,
    medium_radius=0.5,
    electrode_radius=0.005,
    current=3.0,
    conductivity=1.0,
    voltage_scale=1,
):
    electrode_potential = voltage_scale * (
        current / (2 * np.pi * electrode_radius * conductivity)
    )
    x = np.linspace(-medium_radius, medium_radius, grid_size)
    y = np.linspace(-medium_radius, medium_radius, grid_size)
    X, Y = np.meshgrid(x, y)
    V = np.zeros((grid_size, grid_size))
    electrode_positions = [(-medium_radius / 3, 0), (medium_radius / 3, 0)]
    for pos in electrode_positions:
        distance = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2)
        electrode_mask = distance <= electrode_radius
        V[electrode_mask] = electrode_potential if pos[0] > 0 else -electrode_potential
    for _ in range(3000):
        V_old = V.copy()
        V[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2])
        for pos in electrode_positions:
            distance = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2)
            electrode_mask = distance <= electrode_radius
            V[electrode_mask] = (
                electrode_potential if pos[0] > 0 else -electrode_potential
            )
        if np.max(np.abs(V - V_old)) < 1e-5:
            break
    return X, Y, V


# Parameters and solving for potential
voltage_scale = 1
X_2d_small, Y_2d_small, V_2d_small = solve_potential_2d_medium(
    electrode_radius=0.050, voltage_scale=voltage_scale
)
X_2d_large, Y_2d_large, V_2d_large = solve_potential_2d_medium(
    electrode_radius=0.025, voltage_scale=voltage_scale
)

# Calculate current density and its magnitude
Ey_small, Ex_small = np.gradient(-V_2d_small)
Ey_large, Ex_large = np.gradient(-V_2d_large)
Jx_small = Ex_small * 1.0
Jy_small = Ey_small * 1.0
Jx_large = Ex_large * 1.0
Jy_large = Ey_large * 1.0
J_magnitude_small = np.sqrt(Jx_small**2 + Jy_small**2)
J_magnitude_large = np.sqrt(Jx_large**2 + Jy_large**2)

# Calculate electric field components
Ex_small_scaled = Ex_small / voltage_scale
Ey_small_scaled = Ey_small / voltage_scale
Ex_large_scaled = Ex_large / voltage_scale
Ey_large_scaled = Ey_large / voltage_scale

# Find global minimum and maximum of potential for consistent scaling
v_min = min(np.min(V_2d_small), np.min(V_2d_large))
v_max = max(np.max(V_2d_small), np.max(V_2d_large))


# Plotting with adjusted layout and improved annotations
plt.figure(figsize=(18, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Plotting potential for 50mm diameter electrodes
plt.subplot(2, 3, 1)
contour50 = plt.contourf(
    X_2d_small,
    Y_2d_small,
    V_2d_small,
    levels=np.linspace(v_min, v_max, 50),
    cmap="RdBu",
)
plt.colorbar(contour50, label="Potential (V/m)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Potential with 50mm Electrodes")
plt.axis("equal")

# Plotting current density streamlines for 50mm diameter electrodes with annotations
plt.subplot(2, 3, 2)
plt.streamplot(X_2d_small, Y_2d_small, Jx_small, Jy_small, color="#FF9999", density=2)
for point in [(50, 50), (30, 70), (70, 30)]:
    plt.plot(X_2d_small[point], Y_2d_small[point], "wo")  # Mark the point
    plt.annotate(
        f"{J_magnitude_small[point]:.2e} A/m²",
        xy=(X_2d_small[point], Y_2d_small[point]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="center",
        color="black",
    )
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Current Density with 50mm Electrodes")
plt.axis("equal")

# Plotting electric field for 50mm diameter electrodes with color based on magnitude
plt.subplot(2, 3, 3)
electric_field_magnitude_small = np.sqrt(Ex_small_scaled**2 + Ey_small_scaled**2)
plt.quiver(
    X_2d_small[::5, ::5],
    Y_2d_small[::5, ::5],
    Ex_small_scaled[::5, ::5],
    Ey_small_scaled[::5, ::5],
    electric_field_magnitude_small[::5, ::5],  # Use magnitude for color
    scale=30,
    cmap="viridis",  # Use a colormap for coloring vectors
)
plt.colorbar(
    label="Electric Field Magnitude (V/m)",
)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Electric Field with 50mm Electrodes")
plt.axis("equal")

# Plotting potential for 25mm diameter electrodes
plt.subplot(2, 3, 4)
contour25 = plt.contourf(
    X_2d_large,
    Y_2d_large,
    V_2d_large,
    levels=np.linspace(v_min, v_max, 50),
    cmap="RdBu",
)
plt.colorbar(contour25, label="Potential (V/m)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Potential with 25mm Electrodes")
plt.axis("equal")

# Plotting current density streamlines for 25mm diameter electrodes with annotations
plt.subplot(2, 3, 5)
plt.streamplot(X_2d_large, Y_2d_large, Jx_large, Jy_large, color="#FF9999", density=2)
for point in [(50, 50), (30, 70), (70, 30)]:
    plt.plot(X_2d_large[point], Y_2d_large[point], "wo")  # Mark the point
    plt.annotate(
        f"{J_magnitude_large[point]:.2e} A/m²",
        xy=(X_2d_large[point], Y_2d_large[point]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="center",
        color="black",
    )
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Current Density with 25mm Electrodes")
plt.axis("equal")

# Plotting electric field for 25mm diameter electrodes with color based on magnitude
plt.subplot(2, 3, 6)
electric_field_magnitude_large = np.sqrt(Ex_large_scaled**2 + Ey_large_scaled**2)
plt.quiver(
    X_2d_large[::5, ::5],
    Y_2d_large[::5, ::5],
    Ex_large_scaled[::5, ::5],
    Ey_large_scaled[::5, ::5],
    electric_field_magnitude_large[::5, ::5],  # Use magnitude for color
    scale=30,
    cmap="viridis",  # Use a colormap for coloring vectors
)
plt.colorbar(label="Electric Field Magnitude (V/m)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Electric Field with 25mm Electrodes")
plt.axis("equal")

plt.tight_layout()
plt.show()
