import pandas as pd
import os

def process_single_maf(maf_file_path, target_genes, filter_silent=True):
    """
    Process a single MAF file and map its mutation status to a predefined gene list.

    Args:
        maf_file_path (str): Full path to a single MAF file.
        target_genes (list or pd.Index): Predefined gene list (union of Hugo_Symbol) for feature alignment.
        filter_silent (bool): Whether to filter out silent mutations.

    Returns:
        pd.DataFrame: A 0/1 mutation feature vector for a single sample (1 row x N genes columns).
                      Returns None if file reading fails or no mutations are present.
    """

    # Define columns to read
    required_cols = ['Hugo_Symbol', 'Tumor_Sample_Barcode']
    if filter_silent:
        required_cols.append('Variant_Classification')

    try:
        # 1. Read data
        # MAF files are usually Tab-delimited and contain comment lines (starting with #)
        df_temp = pd.read_csv(
            maf_file_path,
            sep='\t',
            comment='#',
            usecols=required_cols,
            # Optimization: read only top rows to quickly get sample ID
            nrows=50000  # Assuming a sample file won't exceed 50,000 mutation rows
        )

    except ValueError as e:
        # Catch errors from usecols, usually due to missing required columns in MAF file
        print(f"Warning: File {os.path.basename(maf_file_path)} is missing required columns. Skipping.")
        return None
    except Exception as e:
        # Catch other reading errors
        print(f"Error: Failed to read file {os.path.basename(maf_file_path)}: {e}. Skipping.")
        return None

    if df_temp.empty:
        # If file is empty or becomes empty after reading, return a zero vector
        sample_id = os.path.basename(maf_file_path).replace(".maf", "")  # Assume sample ID is in filename
        return pd.DataFrame(0, index=[sample_id], columns=target_genes)

    # 2. Filter silent mutations
    if filter_silent:
        df_temp = df_temp[df_temp['Variant_Classification'] != 'Silent']

    # 3. Extract sample ID
    # MAF file typically contains one sample ID, take the first non-null value
    sample_id = df_temp['Tumor_Sample_Barcode'].dropna().iloc[0]

    # 4. Build mutation feature
    # Only need the list of mutated genes
    mutated_genes = df_temp['Hugo_Symbol'].unique()

    # 5. Map and align to target feature space

    # Create a Series with target genes as index, initialized to 0
    feature_vector = pd.Series(0, index=target_genes)

    # Set positions of actually mutated genes to 1
    # isin() quickly finds genes from mutated_genes and sets them to 1 in the Series
    feature_vector[feature_vector.index.isin(mutated_genes)] = 1

    # Convert to DataFrame (1 row x N columns) and name index with sample ID
    result_df = feature_vector.to_frame().T
    result_df.index = [sample_id]

    return result_df
