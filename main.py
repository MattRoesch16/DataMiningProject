import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# BEGINNING OF PREPROCESSING SECTION

def load_csv():
    df1 = pd.read_excel("Predictions.xlsx")
    df1.head()

    df2 = pd.read_excel("oasis_longitudinal_demographics.xlsx")
    df2.head()

    return df1, df2


def find_unknowns(df):
    df = df.replace('?', np.NaN)
    print('Number of missing values:')
    for col in df.columns:
        print('\t%s: %d' % (col, df[col].isna().sum()))
    return df


def replace_with_removal(df):
    print('Number of rows in original data = %d' % (df.shape[0]))
    df = df.dropna()
    print('Number of rows after discarding missing values = %d' % (df.shape[0]))
    return df

def replace_with_mean(df):
    for i in df:
        if i == "SES" or i == "MMSE":
            mean_value = df[i].mean()
            print("Replacing NaN values of", i, "with mean of:", mean_value)
            df[i].fillna(value=mean_value, inplace=True)
    return df


def drop_unneeded(df):
    for i in df:
        if i == "Hand":
            print("Dropping column:", i)
            df = df.drop(["Hand"], axis=1)
    return df


def drop_dups(df):
    print('Number of rows in original data = %d' % (df.shape[0]))
    df = df.drop_duplicates()
    print('Number of rows after discarding duplicated values = %d' % (df.shape[0]))
    return df

# END OF PREPROCESSING SECTION


# BEGINNING OF ANALYSIS SECTION
    # PropQuestion
def Proposed_Question_1(df):
    # Load  dataset
    #dataset_path = "oasis_longitudinal_demographics.xlsx"
    #df = pd.read_excel(dataset_path)

    # Select relevant columns for clustering
    selected_columns = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    data_for_clustering = df[selected_columns]

    # Handling missing values
    #data_for_clustering = data_for_clustering.fillna(data_for_clustering.mean())

    # Standardize  data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=2, random_state=1)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    df['PCA1'] = principal_components[:, 0]
    df['PCA2'] = principal_components[:, 1]

    # Visualize the clustering results
    plt.figure(figsize=(10, 6))
    colors = np.array(['blue', 'red'])
    plt.scatter(df['PCA1'], df['PCA2'], c=colors[df['Cluster']], alpha=0.5)
    plt.title('Clustering of Subjects Based on Brain Patterns')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig("ProposedQuestion#1")

    # Assuming pca is the PCA model fitted on the data
    loadings = pca.components_

    # Access loadings for PC1 and PC2
    loading_pc1 = loadings[0, :]  # Loadings for PC1
    loading_pc2 = loadings[1, :]  # Loadings for PC2

    # Print the loadings and corresponding feature names
    for feature, loading_1, loading_2 in zip(selected_columns, loading_pc1, loading_pc2):
        print(f"{feature}: Loading for PC1={loading_1:.3f}, Loading for PC2={loading_2:.3f}")

# END OF ANALYSIS SECTION


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # BEGINNING OF PREPROCESSING SECTION
    predictions, demographics = load_csv()
    predictions = find_unknowns(predictions)
    demographics = find_unknowns(demographics)
    demographics = replace_with_mean(demographics)
    predictions = drop_unneeded(predictions)
    demographics = drop_unneeded(demographics)
    predictions = drop_dups(predictions)
    demographics = drop_dups(demographics)

    print()
    print("Predicitons: ")
    print(predictions)
    print()
    print("oasis_longitudinal_demographics: ")
    print(demographics)
    print()
    print("Preprocessing Finished")
    # END OF PREPROCESSING SECTION

    # BEGINNING OF ANALYSIS SECTION
    Proposed_Question_1(demographics)
    # END OF ANALYSIS SECTION


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
