import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# ======================
# CONFIGURATION
# ======================
grid_size = 6
max_water = 5.0
decay_amount = 0.75
decay_interval_seconds = 5
frame_interval_ms = 100
frames_per_second = 1000 / frame_interval_ms
decay_interval_frames = int(decay_interval_seconds * frames_per_second)
refill_amount = max_water

# Vehicle tank
vehicle_max_water = 10.0
vehicle_water = vehicle_max_water
low_vehicle_water_threshold = 2.0
water_usage_per_refill = 2.0
movement_water_loss = 0.02

# Water tank location (grid coords)
tank_pos = (0, 0)
fill_threshold = 1.5

# ======================
# INITIAL WATER LEVELS
# ======================
np.random.seed(42)
water_levels = np.random.uniform(1, max_water, (grid_size, grid_size))

# Vehicle state
vehicle_pos = [0, 0]
target_pos = None
path = []
frame_count = 0

# Smooth animation vars
move_progress = 0.0
current_segment_start = None
current_segment_end = None
movement_speed = 0.2

# ======================
# FULL GRID FOR DISPLAY
# ======================
full_grid_size = 2 * grid_size + 1
full_water_grid = np.full((full_grid_size - 1, full_grid_size - 1), np.nan)

def update_full_water_grid():
    for i in range(grid_size):
        for j in range(grid_size):
            full_water_grid[2*i, 2*j] = water_levels[i, j]

update_full_water_grid()

# ======================
# MATPLOTLIB SETUP
# ======================
fig, ax = plt.subplots()
plt.title("Water Levels & Smart Refill Vehicle (Dynamic Pathfinding)", color="white")

# Black background
ax.set_facecolor("black")
fig.patch.set_facecolor("black")

x = np.arange(0, 2 * grid_size + 1, 1)
y = np.arange(0, 2 * grid_size + 1, 1)

mesh = ax.pcolormesh(
    x, y, full_water_grid,
    cmap="coolwarm_r",
    vmin=0, vmax=max_water,
    edgecolors="gray", linewidth=1
)

# Pulsing tank patch (bigger size)
tank_patch = plt.Rectangle(
    (tank_pos[1] * 2 - 0.2, tank_pos[0] * 2 - 0.2),
    2.4, 2.4,
    facecolor="#00BFFF", edgecolor="white", lw=2, zorder=3
)
ax.add_patch(tank_patch)

cbar = plt.colorbar(mesh, ax=ax)
cbar.set_label('Crop Water Level', color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

vehicle_marker, = ax.plot([], [], 'o', markersize=12, label='Refill Vehicle')

ax.set_xlim(-1, 2 * grid_size)
ax.set_ylim(-1, 2 * grid_size)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.legend(loc='upper right', facecolor="black", labelcolor="white")

# ======================
# PATHFINDING (A*)
# ======================
def find_lowest_water_cells():
    low_cells = np.argwhere(water_levels <= np.min(water_levels))
    return [tuple(c) for c in low_cells]

def grid_to_road_coords(cell):
    return [2 * cell[0], 2 * cell[1]]

# Allow 8 directions
directions = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
    (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))
]

def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def find_path(start, goals):
    goal = goals[0]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {tuple(start): 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if tuple(current) == tuple(goal):
            path = []
            while tuple(current) in came_from:
                path.append(current)
                current = came_from[tuple(current)]
            path.reverse()
            return path

        for dr, dc, cost in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < full_grid_size and 0 <= nc < full_grid_size:
                tentative_g = g_score[tuple(current)] + cost
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + heuristic((nr, nc), goal)
                    heapq.heappush(open_set, (f_score, [nr, nc]))
    return None

# ======================
# UPDATE FUNCTION
# ======================
def update(frame):
    global water_levels, vehicle_pos, target_pos, path, vehicle_water
    global frame_count, move_progress, current_segment_start, current_segment_end

    frame_count += 1

    # Periodic decay
    if frame_count % decay_interval_frames == 0 and not path:
        water_levels = np.maximum(0, water_levels - decay_amount)

    update_full_water_grid()
    mesh.set_array(full_water_grid.ravel())

    # If at crop location
    if vehicle_pos[0] % 2 == 0 and vehicle_pos[1] % 2 == 0:
        grid_cell = (vehicle_pos[0] // 2, vehicle_pos[1] // 2)
        if grid_cell != tank_pos and water_levels[grid_cell] <= fill_threshold and vehicle_water > 0:
            water_levels[grid_cell] = max_water
            vehicle_water = max(0, vehicle_water - water_usage_per_refill)

    # At tank
    if vehicle_pos == grid_to_road_coords(tank_pos):
        vehicle_water = vehicle_max_water

    # Decide target
    if vehicle_water <= low_vehicle_water_threshold:
        if target_pos != tank_pos:
            new_path = find_path(vehicle_pos, [grid_to_road_coords(tank_pos)])
            if new_path:
                path = new_path
                target_pos = tank_pos
                current_segment_start = None
    else:
        lowest_cells = find_lowest_water_cells()
        if lowest_cells:
            worst_tank = lowest_cells[0]
            worst_level = water_levels[worst_tank]
            if target_pos is None or water_levels[target_pos] > worst_level:
                new_path = find_path(vehicle_pos, [grid_to_road_coords(worst_tank)])
                if new_path:
                    path = new_path
                    target_pos = worst_tank
                    current_segment_start = None

    # Move
    if current_segment_start is None and path:
        current_segment_start = vehicle_pos
        current_segment_end = path.pop(0)
        move_progress = 0.0

    if current_segment_start is not None:
        move_progress += movement_speed
        vehicle_water = max(0, vehicle_water - movement_water_loss)
        if move_progress >= 1.0:
            vehicle_pos = current_segment_end
            current_segment_start = None
            current_segment_end = None
            move_progress = 0.0
            if not path:
                target_pos = None

    # Smooth position
    if current_segment_start is not None:
        r = current_segment_start[0] + (current_segment_end[0] - current_segment_start[0]) * move_progress
        c = current_segment_start[1] + (current_segment_end[1] - current_segment_start[1]) * move_progress
    else:
        r, c = vehicle_pos

    # ======================
    # Gradual purple → white transition
    # ======================
    purple_rgb = np.array([0.6, 0.0, 1.0])  # neon purple
    white_rgb = np.array([1.0, 1.0, 1.0])   # white

    blend_factor = np.clip(vehicle_water / vehicle_max_water, 0.0, 1.0)
    base_color = purple_rgb * blend_factor + white_rgb * (1 - blend_factor)

    # Low water → slight pulsing
    if vehicle_water <= low_vehicle_water_threshold:
        glow = 0.85 + 0.15 * np.sin(frame / 3)
        base_color = np.clip(base_color * glow, 0, 1)

    vehicle_marker.set_color(base_color)
    vehicle_marker.set_data([c + 0.5], [r + 0.5])

    # Pulsing tank glow (faster)
    glow_factor = 0.5 + 0.5 * np.sin(frame / 4)
    tank_patch.set_facecolor((0.0 * glow_factor, 0.7 + 0.3 * glow_factor, 1.0))

    return mesh, vehicle_marker, tank_patch

# ======================
# RUN
# ======================
ani = FuncAnimation(
    fig, update, frames=1000,
    interval=frame_interval_ms, blit=True, repeat=False
)
plt.show()
