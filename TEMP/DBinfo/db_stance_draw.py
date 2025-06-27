import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Stance Change Trend Visualization Tool')
    parser.add_argument('--input_file', type=str, default='experiments/topic_3_Qwen3-14B_True_20250519_104621/attitude_analysis_results.csv',
                      help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, default='experiments/topic_3_Qwen3-14B_True_20250519_104621',
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
    
    # Define markers and colors for each stance
    stance_styles = {
        'pro': {'marker': 's', 'color': 'blue'},  # square, blue
        'con': {'marker': '^', 'color': 'red'},   # triangle, red
        'neutral': {'marker': 'o', 'color': 'green'}  # circle, green
    }
    
    # Draw three lines with different styles
    for stance in ['pro', 'con', 'neutral']:
        if stance in stance_counts.index:
            style = stance_styles[stance]
            plt.plot(stance_counts.columns, stance_counts.loc[stance], 
                    marker=style['marker'], color=style['color'],
                    label=stance, linewidth=2)
    
    # Set chart properties with larger font sizes
    plt.title('User Stance Change Trend', fontsize=28, pad=20)
    plt.xlabel('Time Step', fontsize=24)
    plt.ylabel('Number of Users', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Stance', fontsize=22, title_fontsize=26, loc='lower left')
    
    # Set x-axis ticks with larger font size
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    
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
