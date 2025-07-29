import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

# Positions (x-coordinates) of points on a line (1D for simplicity)
ambulance_pos = 0
junction_1_pos = 20
junction_2_pos = 50
hospital_pos = 80

speed = 2  # units per second (simulated speed)
signal_advance_time = 10  # seconds before arrival to send signal

# Flags to check if signals sent
signal_sent_j1 = False
signal_sent_j2 = False

fig, ax = plt.subplots()
ax.set_xlim(-5, 85)
ax.set_ylim(-5, 5)
ax.get_yaxis().set_visible(False)
ax.set_title("Ambulance approaching junctions and hospital")

# Plot fixed points
junction1_dot, = ax.plot(junction_1_pos, 0, 'go', markersize=12, label='Junction 1')
junction2_dot, = ax.plot(junction_2_pos, 0, 'go', markersize=12, label='Junction 2')
hospital_dot, = ax.plot(hospital_pos, 0, 'ro', markersize=12, label='Hospital')
ambulance_dot, = ax.plot(ambulance_pos, 0, 'bo', markersize=12, label='Ambulance')

ax.legend(loc='upper left')

start_time = time.time()

def send_signal(junction_name):
    print(f"Signal sent to {junction_name}: Clearing traffic!")
    # Simulate junction clearing traffic here if needed

def update(frame):
    global ambulance_pos, signal_sent_j1, signal_sent_j2
    elapsed = frame  # Using frame as elapsed seconds for simplicity

    # Ambulance moves forward
    ambulance_pos = speed * elapsed

    # Update ambulance position (x and y must be sequences)
    ambulance_dot.set_data([ambulance_pos], [0])

    # Calculate ETA to junction 1 and 2
    eta_j1 = (junction_1_pos - ambulance_pos) / speed
    eta_j2 = (junction_2_pos - ambulance_pos) / speed

    # Send signal 10 seconds before arrival to junction 1
    if not signal_sent_j1 and eta_j1 <= signal_advance_time and ambulance_pos < junction_1_pos:
        send_signal("Junction 1")
        signal_sent_j1 = True
        junction1_dot.set_color('yellow')  # Indicate traffic cleared

    # Send signal 10 seconds before arrival to junction 2
    if not signal_sent_j2 and eta_j2 <= signal_advance_time and ambulance_pos < junction_2_pos:
        send_signal("Junction 2")
        signal_sent_j2 = True
        junction2_dot.set_color('yellow')

    # Stop animation after reaching hospital
    if ambulance_pos >= hospital_pos:
        print("Ambulance reached the hospital!")
        ani.event_source.stop()

    return ambulance_dot, junction1_dot, junction2_dot

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 50), interval=1000, blit=False)
plt.show()
