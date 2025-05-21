import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# File paths
input_file = 'totalzvalues/0hr.xlsx'
output_file = 'z_pca_scores/0hr_pca_by_instance_2.xlsx'

# Banana sheet names
banana_sheets = ['B1', 'B2', 'B3', 'B4','B5','B6','B7','B8','B9','B10']

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Excel writer to collect results
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet in banana_sheets:
        # Load sheet
        df = pd.read_excel(input_file, sheet_name=sheet)

        # Get impedance data: rows = frequencies, columns = I1 to I14
        impedance_data = df.iloc[:, 1:15]  # exclude frequency column

        # Transpose: now each row is one instance (I1 to I14), columns = frequencies
        impedance_transposed = impedance_data.T

        # Standardize across frequencies
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(impedance_transposed)

        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # Save results with instance labels
        instance_labels = impedance_data.columns.tolist()
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        pca_df['Instance'] = instance_labels

        # Reorder columns
        pca_df = pca_df[['Instance', 'PC1', 'PC2']]

        # Write to Excel sheet
        pca_df.to_excel(writer, sheet_name=sheet, index=False)

        # Plot PCA projection
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], c='green')
        for i, label in enumerate(pca_df['Instance']):
            plt.text(pca_df['PC1'][i], pca_df['PC2'][i], label, fontsize=9)
        plt.title(f'PCA of Instances – {sheet}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Explained variance
        print(f"{sheet} - Explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")

print(f"\n✅ PCA results saved by instance to: {output_file}")
