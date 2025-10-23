import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations
import os

def print_stats(leafs):
    """
    Print statistics about the leafs output from df_to_complete_leafs().
    
    Args:
        leafs: List of dictionaries with 'ranges', 'count', and 'initial' keys
    """
    if not leafs:
        print("No leaf data provided")
        return
    
    # Count entries
    total_entries = len(leafs)
    initial_true_entries = sum(1 for leaf in leafs if leaf.get('initial', False))
    initial_false_entries = total_entries - initial_true_entries
    
    # Sum counts
    total_count = sum(leaf.get('count', 0) for leaf in leafs)
    initial_true_count = sum(leaf.get('count', 0) for leaf in leafs if leaf.get('initial', False))
    initial_false_count = total_count - initial_true_count
    
    # Check for overlapping ranges and collect overlapping pairs
    num_dims = len(leafs[0]['ranges']) if leafs else 0
    overlapping_pairs = []
    
    for i in range(len(leafs)):
        for j in range(i + 1, len(leafs)):
            # Check if leafs i and j overlap in all dimensions
            overlap_in_all_dims = True
            for dim in range(num_dims):
                r1_min = leafs[i]['ranges'][dim]['min']
                r1_max = leafs[i]['ranges'][dim]['max']
                r2_min = leafs[j]['ranges'][dim]['min']
                r2_max = leafs[j]['ranges'][dim]['max']
                
                # Check if ranges overlap (not just touch at endpoints)
                if not (r1_min < r2_max and r2_min < r1_max):
                    overlap_in_all_dims = False
                    break
            
            if overlap_in_all_dims:
                overlapping_pairs.append((i, j))
    
    # Count overlapping pairs by initial type
    true_true_pairs = 0
    false_false_pairs = 0
    true_false_pairs = 0
    
    for i, j in overlapping_pairs:
        leaf1_initial = leafs[i].get('initial', False)
        leaf2_initial = leafs[j].get('initial', False)
        
        if leaf1_initial and leaf2_initial:
            true_true_pairs += 1
        elif not leaf1_initial and not leaf2_initial:
            false_false_pairs += 1
        else:
            true_false_pairs += 1
    
    # Calculate range widths statistics
    range_width_stats = {}
    
    for dim in range(num_dims):
        dim_width_stats = {}
        
        for leaf in leafs:
            width = leaf['ranges'][dim]['max'] - leaf['ranges'][dim]['min']
            width = round(width, 10)  # Round to avoid floating point precision issues
            
            if width not in dim_width_stats:
                dim_width_stats[width] = {'item_count': 0, 'total_count': 0}
            
            dim_width_stats[width]['item_count'] += 1
            dim_width_stats[width]['total_count'] += leaf.get('count', 0)
        
        range_width_stats[dim] = dim_width_stats
    
    # Find uncovered spaces for each dimension
    dimension_gaps = []
    
    for dim in range(num_dims):
        # Collect all ranges for this dimension
        ranges = []
        for leaf in leafs:
            range_min = leaf['ranges'][dim]['min']
            range_max = leaf['ranges'][dim]['max']
            ranges.append((range_min, range_max))
        
        # Sort ranges and merge overlapping ones
        ranges = sorted(set(ranges))
        merged_ranges = []
        
        for start, end in ranges:
            if merged_ranges and start <= merged_ranges[-1][1]:
                # Overlapping range, merge it
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
            else:
                # Non-overlapping range, add it
                merged_ranges.append((start, end))
        
        # Find gaps in [0, 1.0]
        gaps = []
        current_pos = 0.0
        
        for start, end in merged_ranges:
            if start > current_pos:
                gaps.append((current_pos, start))
            current_pos = max(current_pos, end)
        
        # Check if there's a gap at the end
        if current_pos < 1.0:
            gaps.append((current_pos, 1.0))
        
        dimension_gaps.append(gaps)
    
    # Print statistics
    print("Leaf Statistics:")
    print("=" * 50)
    print(f"Total entries: {total_entries:,}")
    print(f"  Initial True entries: {initial_true_entries:,}")
    print(f"  Initial False entries: {initial_false_entries:,}")
    print()
    print(f"Total count: {total_count:,}")
    print(f"  Initial True count: {initial_true_count:,}")
    print(f"  Initial False count: {initial_false_count:,}")
    
    # Print overlap information
    print()
    print(f"Overlapping pairs found: {len(overlapping_pairs)}")
    print(f"  True-True pairs: {true_true_pairs}")
    print(f"  False-False pairs: {false_false_pairs}")
    print(f"  True-False pairs: {true_false_pairs}")
    if overlapping_pairs:
        print("Overlapping leaf pairs:")
        pairs_to_show = min(10, len(overlapping_pairs))
        for i in range(pairs_to_show):
            idx1, idx2 = overlapping_pairs[i]
            leaf1_initial = leafs[idx1].get('initial', False)
            leaf2_initial = leafs[idx2].get('initial', False)
            pair_type = f"{'True' if leaf1_initial else 'False'}-{'True' if leaf2_initial else 'False'}"
            print(f"  Pair {i+1} ({pair_type}): Leaf {idx1} and Leaf {idx2}")
            
            # Print first leaf completely
            print(f"    Leaf {idx1}:")
            for dim in range(num_dims):
                r1 = leafs[idx1]['ranges'][dim]
                print(f"      Dim {dim}: [{r1['min']:.6f}, {r1['max']:.6f}]")
            
            # Print second leaf completely
            print(f"    Leaf {idx2}:")
            for dim in range(num_dims):
                r2 = leafs[idx2]['ranges'][dim]
                print(f"      Dim {dim}: [{r2['min']:.6f}, {r2['max']:.6f}]")
        
        if len(overlapping_pairs) > 10:
            print(f"  ... and {len(overlapping_pairs) - 10} more pairs")
    
    # Print range width statistics
    print()
    print("Range Width Statistics:")
    print("-" * 30)
    for dim in range(num_dims):
        print(f"Dimension {dim}:")
        dim_stats = range_width_stats[dim]
        sorted_widths = sorted(dim_stats.keys())
        for width in sorted_widths:
            stats = dim_stats[width]
            print(f"  Width {width:.6f}: {stats['item_count']} items, total count: {stats['total_count']:,}")
    
    # Print gap information
    if dimension_gaps:
        print()
        print("Uncovered spaces in [0, 1.0]:")
        print("-" * 30)
        for dim, gaps in enumerate(dimension_gaps):
            total_gap_width = sum(end - start for start, end in gaps)
            print(f"Dimension {dim}: {len(gaps)} gaps, total width: {total_gap_width:.6f}")
            if gaps:
                for i, (start, end) in enumerate(gaps):
                    print(f"  Gap {i+1}: [{start:.6f}, {end:.6f}] (width: {end-start:.6f})")
