import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Stance Change Trend Visualization Tool')
    parser.add_argument('--input_file', type=str, default='experiments/topic_4_Qwen3-14B_True_20250516_115712/attitude_analysis_results.csv',
                      help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, default='experiments/topic_4_Qwen3-14B_True_20250516_115712',
                      help='Output image directory')
    parser.add_argument('--dpi', type=int, default=300,
                      help='Output image DPI')
    return parser.parse_args()

def draw_stance_trend(df, output_path, dpi=300):
    # Calculate stance counts for each time point
    stance_counts = pd.DataFrame()
    for col in df.columns:
        if col != 'user_id':
            counts = df[col].value_counts()
            stance_counts[col] = counts
    
    # Fill missing values (if a stance doesn't appear at a time point)
    stance_counts = stance_counts.fillna(0)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Draw three lines
    for stance in ['pro', 'con', 'neutral']:
        if stance in stance_counts.index:
            plt.plot(stance_counts.columns, stance_counts.loc[stance], 
                    marker='o', label=stance, linewidth=2)
    
    # Set chart properties
    plt.title('User Stance Change Trend', fontsize=14, pad=15)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Stance', fontsize=10)
    
    # Set x-axis ticks
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Read CSV file
    df = pd.read_csv(args.input_file)
    
    # Create output directory (if it doesn't exist)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set output file path
    output_path = os.path.join(args.output_dir, 'stance_trend.png')
    
    # Draw trend chart
    draw_stance_trend(df, output_path, args.dpi)
    print(f"Trend chart saved to: {output_path}")

if __name__ == "__main__":
    main()
