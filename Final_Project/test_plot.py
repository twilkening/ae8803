import matplotlib.pyplot as plt

# Turn on interactive mode
plt.ion()

# Create a figure and axes
fig, ax = plt.subplots()

# Example of plotting some initial data
ax.plot([1, 2, 3], [1, 4, 9], "r-")  # Initial plot

# Updating the plot within some data refresh loop
for i in range(10):
    # Update data here, just adding a new point for demonstration
    ax.plot([1, 2, 3, 3 + i], [1, 4, 9, 9 + i], "r-")
    # Refresh the display
    fig.canvas.draw()
    fig.canvas.flush_events()

# To make changes persist after the loop, you may disable interactive mode using plt.ioff()
plt.ioff()
