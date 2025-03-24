import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
import sys


def parse_cpp_output(text):
    """
    Parse the printed output from the C++ program.
    
    It looks for sections that start with the line:
        Idle | Scan | Memory | Merge | Push
    and then reads the subsequent lines (until a separator line starting with '=')
    as bucket timing data.
    
    Returns a list of blocks, each block being a dict with key 'buckets' 
    which is a list of bucket dictionaries. Each bucket dictionary has the keys:
    'idle', 'scanning', 'memory_alloc', 'merging', and 'push'.
    """
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for the header line indicating bucket data.
        if line == "Idle | Scan | Memory | Merge | Push":
            # Prepare a new block
            buckets = []
            i += 1
            # Read until we hit a separator line (which starts with '=')
            while i < len(lines) and not lines[i].strip().startswith("===") and lines[i].strip():
                # Each bucket line is expected to have 5 fields separated by '|'
                parts = [p.strip() for p in lines[i].split("|")]
                if len(parts) == 5:
                    try:
                        idle = float(parts[0])
                        scan = float(parts[1])
                        memory_alloc = float(parts[2])
                        merging = float(parts[3])
                        push = float(parts[4])
                    except ValueError:
                        # If conversion fails, skip the line.
                        i += 1
                        continue
                    bucket = {
                        'idle': idle,
                        'scanning': scan,
                        'memory_alloc': memory_alloc,
                        'merging': merging,
                        'push': push
                    }
                    buckets.append(bucket)
                i += 1
            # Append this block only if we parsed some buckets.
            if buckets:
                blocks.append({'buckets': buckets})
        else:
            i += 1
    return blocks


def plot_blocks(core_graphs):
    """
    Create a horizontal stacked bar chart where each bar represents a block.
    For each block, the buckets are concatenated in order (idle → scan → memory → merge → push).
    The x-axis is set based on the maximum total time across all blocks,
    and margins are removed so the bars are super stretched horizontally.
    """
    # Define the order of metrics per bucket.
    metric_order = ['idle', 'scanning', 'memory_alloc', 'merging', 'push']
    # Human-friendly names for the legend.
    metric_names = {
        'idle': 'Idle',
        'scanning': 'Scan',
        'memory_alloc': 'Memory Alloc',
        'merging': 'Merge',
        'push': 'Push'
    }
    
    # Compute the maximum total time across all blocks.
    max_total = 0
    for block in core_graphs:
        # Sum all segments for all buckets in the block.
        block_total = sum(bucket[metric] for bucket in block['buckets'] for metric in metric_order)
        max_total = max(max_total, block_total)
    
    # Set up the plot with a super stretched horizontal dimension.
    # Here, we use a very wide figure size (40 inches wide) and a fixed height (10 inches).
    fig, ax = plt.subplots(figsize=(40, 10))
    
    # Use the default color cycle from matplotlib.
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_for_metric = {metric: color_cycle[i % len(color_cycle)] for i, metric in enumerate(metric_order)}
    
    # To avoid duplicate legend entries, track which metric labels have been added.
    added_label = {metric: False for metric in metric_order}
    
    # Plot each block as one horizontal bar.
    for block_idx, block in enumerate(core_graphs):
        # Create a list of segments for the block by concatenating each bucket's metrics.
        segments = []
        for bucket in block['buckets']:
            for metric in metric_order:
                segments.append((metric, bucket[metric]))
    
        # Plot the segments stacked horizontally.
        left = 0
        for metric, value in segments:
            if not added_label[metric]:
                ax.barh(block_idx, value, left=left, color=colors_for_metric[metric],
                        edgecolor='black', label=metric_names[metric])
                added_label[metric] = True
            else:
                ax.barh(block_idx, value, left=left, color=colors_for_metric[metric],
                        edgecolor='black')
            left += value
    
    # Label y-axis with block numbers.
    ax.set_yticks(range(len(core_graphs)))
    ax.set_yticklabels([f'Block {i}' for i in range(len(core_graphs))])
    ax.set_xlabel('Time (s)')
    ax.set_title('Stacked Horizontal Bar Chart per Block\n(Buckets Stacked in Order: Idle → Scan → Memory → Merge → Push)')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # Set x-axis limits to fill the entire horizontal span.
    ax.set_xlim(0, max_total)
    # Remove extra horizontal margins.
    ax.margins(x=0)
    
    plt.tight_layout()
    plt.savefig('output.png')


def plot_blocks_optimized(core_graphs, block_spacing=0.1):
    """
    Create a horizontal stacked bar chart using PatchCollection to speed up plotting.
    Each bar represents a block and within each block, bucket segments are stacked in order:
    idle → scan → memory → merge → push.
    
    The vertical spacing between blocks is controlled by the block_spacing parameter.
    """
    # Define the order of metrics per bucket and human-friendly names.
    metric_order = ['idle', 'scanning', 'memory_alloc', 'merging', 'push']
    metric_names = {
        'idle': 'Idle',
        'scanning': 'Scan',
        'memory_alloc': 'Memory Alloc',
        'merging': 'Merge',
        'push': 'Push'
    }
    
    # Compute maximum total time across all blocks.
    max_total = 0
    for block in core_graphs:
        block_total = sum(bucket[metric] for bucket in block['buckets'] for metric in metric_order)
        max_total = max(max_total, block_total)
    
    # Determine figure size based on number of blocks and block_spacing.
    num_blocks = len(core_graphs)
    fig_height = max(num_blocks * block_spacing, 10)  # ensure a minimum height of 10 inches
    fig, ax = plt.subplots(figsize=(40, fig_height))
    
    # Use the default color cycle.
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_for_metric = {metric: color_cycle[i % len(color_cycle)] for i, metric in enumerate(metric_order)}
    
    # Create rectangle patches for each segment.
    patches_list = []
    for block_idx, block in enumerate(core_graphs):
        # Determine vertical position: center each block at block_idx * block_spacing.
        y_center = block_idx * block_spacing
        # Use a bar height smaller than block_spacing so there is visible space.
        bar_height = block_spacing * 0.8
        y_bottom = y_center - bar_height / 2
        
        left = 0
        for bucket in block['buckets']:
            for metric in metric_order:
                width = bucket[metric]
                rect = patches.Rectangle((left, y_bottom), width, bar_height)
                rect.set_facecolor(colors_for_metric[metric])
                rect.set_edgecolor('black')
                patches_list.append(rect)
                left += width  # move to the next segment
    
    # Add all patches at once.
    collection = PatchCollection(patches_list, match_original=True)
    ax.add_collection(collection)
    
    # Configure axes: set limits and y-ticks spaced by block_spacing.
    ax.set_xlim(0, max_total)
    ax.set_ylim(-block_spacing/2, (num_blocks - 0.5) * block_spacing)
    ax.set_yticks([i * block_spacing for i in range(num_blocks)])
    ax.set_yticklabels([f'Block {i}' for i in range(num_blocks)])
    ax.set_xlabel('Time (s)')
    ax.set_title('Stacked Horizontal Bar Chart per Block\n(Buckets Stacked in Order: Idle → Scan → Memory → Merge → Push)')
    
    # Create a custom legend.
    legend_handles = [
        mlines.Line2D([], [], color=colors_for_metric[m], marker='s', linestyle='None', markersize=10, label=metric_names[m])
        for m in metric_order
    ]
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # Remove extra horizontal margins.
    ax.margins(x=0)
    
    plt.tight_layout()

    # make it horizontally long
    plt.gcf().set_size_inches(80, 10)

    plt.savefig('output-fast.png')


if __name__ == "__main__":
    # take argument from the command line
    if len(sys.argv) != 2:
        print("Usage: python test.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        text = f.read()

    print("Parsing output...")

    core_graphs = parse_cpp_output(text)
    print("Plotting output...")
    # plot_blocks(core_graphs)
    plot_blocks_optimized(core_graphs)
    print("Output saved to 'output.png'")