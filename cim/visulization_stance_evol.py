#!/usr/bin/env python3
"""
Stance Evolution Trend Visualization
Analyze user stance changes over timesteps and create line charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import os

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class StanceEvolutionVisualizer:
    """Stance Evolution Visualizer"""
    
    def __init__(self, csv_path: str):
        """
        Initialize visualizer
        
        Args:
            csv_path: Path to stance evolution data CSV file
        """
        self.csv_path = csv_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load data"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✓ Successfully loaded data with {len(self.data)} records")
            print(f"✓ Timestep range: {self.data['timestep'].min()} - {self.data['timestep'].max()}")
            print(f"✓ Number of users: {self.data['user_id'].nunique()}")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            self.data = None
    
    def get_stance_distribution_over_time(self) -> pd.DataFrame:
        """
        Get stance distribution over timesteps
        
        Returns:
            DataFrame containing stance counts for each timestep
        """
        if self.data is None:
            return pd.DataFrame()
        
        # Group by timestep and stance
        stance_dist = self.data.groupby(['timestep', 'stance']).size().reset_index(name='count')
        
        # Pivot table to make each stance a column
        stance_pivot = stance_dist.pivot(index='timestep', columns='stance', values='count').fillna(0)
        
        return stance_pivot
    
    def get_user_stance_changes(self) -> Dict:
        """
        Analyze user stance changes
        
        Returns:
            User stance change statistics
        """
        if self.data is None:
            return {}
        
        changes = {}
        for user_id in self.data['user_id'].unique():
            user_data = self.data[self.data['user_id'] == user_id].sort_values('timestep')
            
            # Calculate stance change count
            stance_changes = 0
            previous_stance = None
            change_points = []
            
            for _, row in user_data.iterrows():
                current_stance = row['stance']
                if previous_stance is not None and current_stance != previous_stance:
                    stance_changes += 1
                    change_points.append({
                        'timestep': row['timestep'],
                        'from': previous_stance,
                        'to': current_stance
                    })
                previous_stance = current_stance
            
            changes[user_id] = {
                'total_changes': stance_changes,
                'change_points': change_points,
                'initial_stance': user_data.iloc[0]['stance'],
                'final_stance': user_data.iloc[-1]['stance'],
                'stance_evolution': user_data['stance'].tolist()
            }
        
        return changes
    
    def plot_stance_distribution_over_time(self, save_path: str = "data/stance_distribution_over_time.png"):
        """
        Plot stance distribution over timesteps
        
        Args:
            save_path: Save path
        """
        if self.data is None:
            print("❌ No data to plot")
            return
        
        stance_dist = self.get_stance_distribution_over_time()
        
        # Convert Chinese stance labels to English
        stance_dist_english = stance_dist.copy()
        stance_dist_english.columns = [self._convert_stance_to_english(col) for col in stance_dist_english.columns]
        
        plt.figure(figsize=(12, 8))
        
        # Plot stacked area chart
        stance_dist_english.plot(kind='area', stacked=True, alpha=0.7)
        
        plt.title('User Stance Distribution Over Timesteps', fontsize=16, fontweight='bold')
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.legend(title='Stance Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save image
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Stance distribution chart saved to: {save_path}")
        plt.show()
    
    def _convert_stance_to_english(self, stance: str) -> str:
        """
        Convert Chinese stance to English
        
        Args:
            stance: Chinese stance string
            
        Returns:
            English stance string
        """
        stance_mapping = {
            '反对': 'Con',
            '中立': 'Neutral', 
            '混合': 'Con',
            '支持': 'Pro',
            '未知': 'Unknown'
        }
        return stance_mapping.get(stance, stance)
    
    def plot_individual_user_evolution(self, user_ids: List[int] = None, 
                                     save_path: str = "data/individual_user_evolution.png"):
        """
        Plot individual user stance evolution
        
        Args:
            user_ids: List of user IDs to plot, if None plot all users
            save_path: Save path
        """
        if self.data is None:
            print("❌ No data to plot")
            return
        
        if user_ids is None:
            user_ids = self.data['user_id'].unique()[:10]  # Default plot first 10 users
        
        # Create stance to value mapping
        stance_mapping = {'反对': 0, '中立': 1, '混合': 0, '支持': 2, '未知': -1}
        
        plt.figure(figsize=(15, 10))
        
        for user_id in user_ids:
            user_data = self.data[self.data['user_id'] == user_id].sort_values('timestep')
            if len(user_data) == 0:
                continue
            
            # Convert stance to numeric values
            stance_values = [stance_mapping.get(stance, -1) for stance in user_data['stance']]
            
            plt.plot(user_data['timestep'], stance_values, 
                    marker='o', linewidth=2, markersize=6, 
                    label=f'User {user_id}')
        
        # Set y-axis labels
        plt.yticks([0, 1, 2], ['Con', 'Neutral', 'Pro'])
        plt.ylim(-0.5, 2.5)
        
        plt.title('Individual User Stance Evolution Trajectories', fontsize=16, fontweight='bold')
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Stance', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save image
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Individual user evolution chart saved to: {save_path}")
        plt.show()
    
    def plot_stance_change_heatmap(self, save_path: str = "data/stance_change_heatmap.png"):
        """
        Plot stance change heatmap
        
        Args:
            save_path: Save path
        """
        if self.data is None:
            print("❌ No data to plot")
            return
        
        # Create stance change matrix
        stances = ['反对', '中立', '混合', '支持']
        stance_labels = ['Con', 'Neutral', 'Con', 'Pro']
        change_matrix = np.zeros((len(stances), len(stances)))
        
        for user_id in self.data['user_id'].unique():
            user_data = self.data[self.data['user_id'] == user_id].sort_values('timestep')
            
            for i in range(1, len(user_data)):
                prev_stance = user_data.iloc[i-1]['stance']
                curr_stance = user_data.iloc[i]['stance']
                
                if prev_stance in stances and curr_stance in stances:
                    prev_idx = stances.index(prev_stance)
                    curr_idx = stances.index(curr_stance)
                    change_matrix[prev_idx][curr_idx] += 1
        
        # Remove duplicate Con rows/columns and combine them
        # Since both '反对' and '混合' map to 'Con', we need to handle this
        unique_stances = ['Con', 'Neutral', 'Pro']
        unique_matrix = np.zeros((3, 3))
        
        # Map the original matrix to the simplified one
        stance_to_idx = {'Con': 0, 'Neutral': 1, 'Pro': 2}
        
        for i, stance in enumerate(stances):
            for j, stance2 in enumerate(stances):
                if stance in ['反对', '混合'] and stance2 in ['反对', '混合']:
                    # Both are Con
                    unique_matrix[0, 0] += change_matrix[i, j]
                elif stance in ['反对', '混合'] and stance2 == '中立':
                    # From Con to Neutral
                    unique_matrix[0, 1] += change_matrix[i, j]
                elif stance in ['反对', '混合'] and stance2 == '支持':
                    # From Con to Pro
                    unique_matrix[0, 2] += change_matrix[i, j]
                elif stance == '中立' and stance2 in ['反对', '混合']:
                    # From Neutral to Con
                    unique_matrix[1, 0] += change_matrix[i, j]
                elif stance == '中立' and stance2 == '中立':
                    # From Neutral to Neutral
                    unique_matrix[1, 1] += change_matrix[i, j]
                elif stance == '中立' and stance2 == '支持':
                    # From Neutral to Pro
                    unique_matrix[1, 2] += change_matrix[i, j]
                elif stance == '支持' and stance2 in ['反对', '混合']:
                    # From Pro to Con
                    unique_matrix[2, 0] += change_matrix[i, j]
                elif stance == '支持' and stance2 == '中立':
                    # From Pro to Neutral
                    unique_matrix[2, 1] += change_matrix[i, j]
                elif stance == '支持' and stance2 == '支持':
                    # From Pro to Pro
                    unique_matrix[2, 2] += change_matrix[i, j]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(unique_matrix, annot=True, fmt='g', cmap='YlOrRd',
                   xticklabels=unique_stances, yticklabels=unique_stances)
        
        plt.title('Stance Change Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Target Stance', fontsize=12)
        plt.ylabel('Source Stance', fontsize=12)
        
        # Save image
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Stance change heatmap saved to: {save_path}")
        plt.show()
    
    def plot_stance_statistics(self, save_path: str = "data/stance_statistics.png"):
        """
        Plot stance statistics charts
        
        Args:
            save_path: Save path
        """
        if self.data is None:
            print("❌ No data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall stance distribution
        stance_counts = self.data['stance'].value_counts()
        stance_labels = [self._convert_stance_to_english(stance) for stance in stance_counts.index]
        
        ax1.pie(stance_counts.values, labels=stance_labels, autopct='%1.1f%%')
        ax1.set_title('Overall Stance Distribution', fontweight='bold')
        
        # 2. Average confidence for each stance
        confidence_by_stance = self.data.groupby('stance')['confidence'].mean()
        stance_labels_confidence = [self._convert_stance_to_english(stance) for stance in confidence_by_stance.index]
        
        ax2.bar(range(len(confidence_by_stance)), confidence_by_stance.values)
        ax2.set_title('Average Confidence by Stance', fontweight='bold')
        ax2.set_ylabel('Average Confidence')
        ax2.set_xticks(range(len(confidence_by_stance)))
        ax2.set_xticklabels(stance_labels_confidence)
        
        # 3. User stance change count distribution
        changes = self.get_user_stance_changes()
        change_counts = [info['total_changes'] for info in changes.values()]
        if change_counts:
            ax3.hist(change_counts, bins=range(max(change_counts) + 2), alpha=0.7)
            ax3.set_title('User Stance Change Count Distribution', fontweight='bold')
            ax3.set_xlabel('Number of Changes')
            ax3.set_ylabel('Number of Users')
        
        # 4. Stance distribution trend over timesteps
        stance_dist = self.get_stance_distribution_over_time()
        for stance in stance_dist.columns:
            stance_label = self._convert_stance_to_english(stance)
            ax4.plot(stance_dist.index, stance_dist[stance], marker='o', label=stance_label)
        ax4.set_title('Stance Distribution Trend Over Time', fontweight='bold')
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Number of Users')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Stance statistics chart saved to: {save_path}")
        plt.show()
    
    def generate_summary_report(self, save_path: str = "data/stance_evolution_summary.txt"):
        """
        Generate analysis summary report
        
        Args:
            save_path: Save path
        """
        if self.data is None:
            print("❌ No data to analyze")
            return
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Stance Evolution Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic information
            f.write("1. Basic Data Information\n")
            f.write(f"   - Total records: {len(self.data)}\n")
            f.write(f"   - Number of users: {self.data['user_id'].nunique()}\n")
            f.write(f"   - Timestep range: {self.data['timestep'].min()} - {self.data['timestep'].max()}\n")
            f.write(f"   - Number of timesteps: {self.data['timestep'].nunique()}\n\n")
            
            # Stance distribution
            f.write("2. Overall Stance Distribution\n")
            stance_counts = self.data['stance'].value_counts()
            for stance, count in stance_counts.items():
                percentage = count / len(self.data) * 100
                stance_label = self._convert_stance_to_english(stance)
                f.write(f"   - {stance_label}: {count} times ({percentage:.1f}%)\n")
            f.write("\n")
            
            # User stance change analysis
            f.write("3. User Stance Change Analysis\n")
            changes = self.get_user_stance_changes()
            total_changes = sum(info['total_changes'] for info in changes.values())
            avg_changes = total_changes / len(changes) if changes else 0
            f.write(f"   - Total change count: {total_changes}\n")
            f.write(f"   - Average changes per user: {avg_changes:.2f}\n")
            
            # Find user with most changes
            if changes:
                max_change_user = max(changes.items(), key=lambda x: x[1]['total_changes'])
                f.write(f"   - User with most changes: User {max_change_user[0]} ({max_change_user[1]['total_changes']} changes)\n")
            
            f.write("\n")
            
            # Time trend analysis
            f.write("4. Time Trend Analysis\n")
            stance_dist = self.get_stance_distribution_over_time()
            for stance in stance_dist.columns:
                initial_count = stance_dist[stance].iloc[0]
                final_count = stance_dist[stance].iloc[-1]
                change = final_count - initial_count
                
                stance_label = self._convert_stance_to_english(stance)
                f.write(f"   - {stance_label}: from {initial_count:.0f} to {final_count:.0f} (change: {change:+.0f})\n")
        
        print(f"✓ Analysis summary report saved to: {save_path}")
    
    def create_all_visualizations(self):
        """Create all visualization charts"""
        print("🎨 Starting to create stance evolution visualization charts...")
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Create various charts
        self.plot_stance_distribution_over_time()
        self.plot_individual_user_evolution()
        self.plot_stance_change_heatmap()
        self.plot_stance_statistics()
        self.generate_summary_report()
        
        print("✅ All visualization charts created successfully!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stance Evolution Trend Visualization")
    parser.add_argument("--csv_path", default="./data/stance_detection_results_evolution.csv", 
                       help="Path to stance evolution data CSV file")
    parser.add_argument("--users", type=int, nargs='+', default=None,
                       help="List of user IDs to plot")
    
    args = parser.parse_args()
    
    print("🎨 Stance Evolution Trend Visualization Tool")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = StanceEvolutionVisualizer(args.csv_path)
    
    if visualizer.data is not None:
        # Create all visualization charts
        visualizer.create_all_visualizations()
        
        # If users are specified, create individual user evolution chart
        if args.users:
            visualizer.plot_individual_user_evolution(
                user_ids=args.users,
                save_path="data/specified_users_evolution.png"
            )
    else:
        print("❌ Unable to load data, please check file path")


if __name__ == "__main__":
    main()
