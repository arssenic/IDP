import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Input and output file paths
input_file = 'totalphvalues/0hr_ph.xlsx'
output_file = 'ph_pca_scores/0hr_ph_pca_by_instance_2.xlsx'

# Sheet names corresponding to bananas
banana_sheets = ['B1', 'B2', 'B3', 'B4','B5','B6','B7','B8','B9','B10']

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Create a writer to save results in multiple sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet in banana_sheets:
        # Load the phase data for the banana
        df = pd.read_excel(input_file, sheet_name=sheet)

        # Extract only phase values (excluding the frequency column)
        phase_data = df.iloc[:, 1:15]  # Columns I1 to I14

        # Transpose so rows = instances, columns = frequency points
        phase_transposed = phase_data.T

        # Standardize across frequency points
        scaler = StandardScaler()
        phase_scaled = scaler.fit_transform(phase_transposed)

        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(phase_scaled)

        # Create DataFrame with PC1 and PC2 per instance
        instance_labels = phase_data.columns.tolist()
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        pca_df['Instance'] = instance_labels
        pca_df = pca_df[['Instance', 'PC1', 'PC2']]  # Reorder

        # Save to the corresponding sheet
        pca_df.to_excel(writer, sheet_name=sheet, index=False)

        # Plot the PCA projection
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], color='purple')
        for i, label in enumerate(pca_df['Instance']):
            plt.text(pca_df['PC1'][i], pca_df['PC2'][i], label, fontsize=9)
        plt.title(f'PCA of Phase Instances – {sheet}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Print explained variance for this banana
        print(f"{sheet} - Explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")

print(f"\n✅ PCA phase results saved to: {output_file}")
