import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Read the data
with open('hike-data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Marker mapping by days
marker_map = {3: '^', 4: 's', 5: 'p'}
def get_marker(days):
    return marker_map.get(days, 'o')

# Assign a color to each hike name
palette = sns.color_palette('tab10', n_colors=len(df['name'].unique()))
color_map = dict(zip(df['name'].unique(), palette))

def plot_hikes(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        plt.scatter(
            row[x], row[y],
            marker=get_marker(row['days']),
            color=color_map[row['name']],
            s=100,
            edgecolor='k',
            label=row['name'] if plt.gca().get_legend_handles_labels()[1].count(row['name']) == 0 else ""
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # Custom legend for colors (hike names)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Hike Name", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# Plot 1: Total elevation vs total distance
plot_hikes(
    x='total_distance_km',
    y='total_elevation_m',
    xlabel='Distance (km)',
    ylabel='Elevation Difference (meters)',
    title='Total Elevation vs Total Distance',
    filename='total_elevation_vs_total_distance.png'
)

# Plot 2: Average elevation per day vs average distance per day
plot_hikes(
    x='avg_distance_per_day_km',
    y='avg_elevation_per_day_m',
    xlabel='Avg Distance per Day (km)',
    ylabel='Avg Elevation Difference per Day (meters)',
    title='Avg Elevation per Day vs Avg Distance per Day',
    filename='avg_elevation_per_day_vs_avg_distance_per_day.png'
)

# Third plot: Horizontal bar plot of avg meters elevation per kilometer
df['elev_per_km'] = df['total_elevation_m'] / df['total_distance_km']

plt.figure(figsize=(8, 5))
sns.barplot(
    y='name',
    x='elev_per_km',
    data=df,
    palette=[color_map[name] for name in df['name']]
)
plt.xlabel('Avg meters elevation per kilometer')
plt.ylabel('Hike Name')
plt.title('Average Elevation Gain per Kilometer by Hike')
plt.tight_layout()
plt.savefig('avg_elevation_per_km_by_hike.png', dpi=150)
plt.close()