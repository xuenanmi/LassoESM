import pandas as pd
import random

# Load the dataset
data = pd.read_excel('SOAL5_1_finalized_HVR_SI.xlsx')   # natural cycalse-lasso peptide pairs predicted from RODEO

# Select relevant columns and drop rows with missing values
selected_data = data[['Cyclase sequence', 'Core', 'Acceptor site', 'Acceptor residue']]
df = selected_data.dropna().drop_duplicates().reset_index(drop=True)

# Print the shape of the cleaned dataset and the number of unique cyclase sequences
print(f"Shape of the dataset after dropping NA and duplicates: {df.shape}")
print(f"Number of unique cyclase sequences: {len(set(df['Cyclase sequence']))}")

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=12).reset_index(drop=True)

# Initialize a list to store the mismatched pairs
mismatched_pairs = []

# Generate mismatched pairs
for index, row in df.iterrows():
    original_cyclase = row['Cyclase sequence']
    original_residue = row['Acceptor residue']
    
    # Find all mismatched peptides
    mismatched_pool = df[(df['Cyclase sequence'] != original_cyclase) & 
                         (df['Acceptor residue'] != original_residue)]
    
    # Randomly select one mismatched peptide
    if not mismatched_pool.empty:
        selected_mismatch = mismatched_pool.sample(n=1).iloc[0]
        mismatched_pairs.append({
            'Cyclase sequence': original_cyclase,
            'Original Core': row['Core'],
            'Original Acceptor residue': original_residue,
            'Mismatched Core': selected_mismatch['Core'],
            'Mismatched Acceptor residue': selected_mismatch['Acceptor residue']
        })

# Convert the list of mismatched pairs into a DataFrame
mismatched_df = pd.DataFrame(mismatched_pairs)

# Check for duplicates in mismatched_df
duplicate_count = mismatched_df.duplicated(subset=['Cyclase sequence', 'Mismatched Core'], keep=False).sum()
print(f"Number of duplicate pairs in mismatched_df: {duplicate_count}")

# Remove duplicates
mismatched_df = mismatched_df.drop_duplicates(subset=['Cyclase sequence', 'Mismatched Core'])

# Merge to remove pairs that are present in the original matched_df
combined_df = mismatched_df.merge(df, left_on=['Cyclase sequence', 'Mismatched Core', 'Mismatched Acceptor residue'],
                                  right_on=['Cyclase sequence', 'Core', 'Acceptor residue'],
                                  how='left', indicator=True)

# Keep only mismatched pairs not present in the original matched_df
filtered_mismatched_df = combined_df[combined_df['_merge'] == 'left_only'].drop(columns=['Core', 'Acceptor residue', '_merge'])

# Rename columns for clarity
negative_df = filtered_mismatched_df[['Cyclase sequence', 'Mismatched Core']]
negative_df = negative_df.rename(columns={'Mismatched Core': 'Core'})

# Save the filtered negative pairs to a CSV file
filtered_mismatched_df.to_csv('negative_pairs_with_filter.csv', index=False)

# Combine with positive pairs
positive_df = df[['Cyclase sequence', 'Core']]
combined_data = pd.concat([positive_df, negative_df], ignore_index=True)

# Add labels: 1 for positive pairs, 0 for negative pairs
combined_data['label'] = [1] * positive_df.shape[0] + [0] * negative_df.shape[0]

# Save the combined dataset to a CSV file
combined_data.to_csv('Cyclase_substrate_pairs_pos_neg_with_filter.csv', index=False)

# Print the shape of the final combined dataset
print(f"Shape of the final combined dataset: {combined_data.shape}")

