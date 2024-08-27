import pandas as pd
import random

data = pd.read_excel('SOAL5_1_finalized_HVR_SI.xlsx')
selected_data = data[['Cyclase sequence','Core','Acceptor site','Acceptor residue']]
selected_data1 = selected_data.dropna(how = 'any')

df = selected_data1.drop_duplicates().reset_index(drop=True)
print(df.shape)
# Shuffle the entire DataFrame before generating negative pairs
df = df.sample(frac=1, random_state=12).reset_index(drop=True)

# Function to generate negative pairs 
def generate_negative_pairs(df):
    negative_pairs = []
    used_cores = set()  # To keep track of used core sequences

    # Shuffle the 'Core' column to create potential negative pairs
    shuffled_cores = df['Core'].sample(frac=1, random_state=42).reset_index(drop=True)

    for idx, row in df.iterrows():
        original_cyclase = row['Cyclase sequence']
        original_core = row['Core']
        original_acceptor_residue = row['Acceptor residue']

        # Iterate through shuffled cores until finding an unused core with a different acceptor residue
        for i in range(len(shuffled_cores)):
            candidate_core = shuffled_cores.iloc[i]
            candidate_acceptor_residue = df[df['Core'] == candidate_core]['Acceptor residue'].values[0]

            if candidate_core not in used_cores and original_acceptor_residue != candidate_acceptor_residue:
                # Create the negative pair
                negative_pair = (original_cyclase, candidate_core)
                
                # Ensure there's no overlap with positive pairs
                if not ((df['Cyclase sequence'] == original_cyclase) & (df['Core'] == candidate_core)).any():
                    negative_pairs.append({
                        'Cyclase sequence': original_cyclase,
                        'Core': candidate_core,
                        'Acceptor site': row['Acceptor site'],  # Keeping original acceptor site
                        'Acceptor residue': candidate_acceptor_residue
                    })
                    used_cores.add(candidate_core)  # Mark the core as used
                break

    # Convert the negative pairs list to a DataFrame
    negative_df = pd.DataFrame(negative_pairs)

    return negative_df

negative_df = generate_negative_pairs(df)

print(negative_df)

# Save the negative pairs to a CSV file
negative_df.to_csv('negative_pairs_with_filter.csv', index=False)

# Combine with positive pairs
combined_data = pd.concat([df.iloc[:, :2], negative_df.iloc[:, :2]], ignore_index=True)
print(combined_data.shape)

combined_data['label'] = df.shape[0] * [1] + negative_df.shape[0] * [0]
combined_data.to_csv('Cyclase_substrate_pairs_pos_neg_with_filter.csv', index=False)
