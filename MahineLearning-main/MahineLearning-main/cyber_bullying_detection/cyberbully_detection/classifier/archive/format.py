# import pandas as pd
# import json

# # Load JSON data
# with open('Dataset.json', 'r') as f:
#     json_data = json.load(f)

# # Extract only "content" and "label" columns
# data = []
# for item in json_data:
#     content = item.get('content', '')
#     label = item.get('annotation', {}).get('label', [''])[0]
#     data.append({'content': content, 'label': label})

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save to CSV
# df.to_csv('dataset.csv', index=False)

# print("CSV file 'dataset.csv' has been created successfully.")
import pandas as pd

# Read the labeled_tweets.csv file
labeled_tweets_df = pd.read_csv('labeled_tweets.csv')

# Rename the columns to match the sus.csv file
labeled_tweets_df = labeled_tweets_df.rename(columns={'full_text': 'comments', 'label': 'tagging'})

# Convert label column to 1 for offensive and 0 for non-offensive
labeled_tweets_df['tagging'] = labeled_tweets_df['tagging'].apply(lambda x: 1 if x == 'offensive' else 0)

# Read the sus.csv file
sus_df = pd.read_csv('sus.csv')

# Concatenate the DataFrames
merged_df = pd.concat([sus_df, labeled_tweets_df[['comments', 'tagging']]])

# Save the merged DataFrame back to sus.csv
merged_df.to_csv('sus.csv', index=False)

