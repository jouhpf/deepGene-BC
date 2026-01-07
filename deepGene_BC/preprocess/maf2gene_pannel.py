import pandas as pd
import os
import glob
import argparse
import sys

def process_maf_files(maf_folder_path, reference_csv_path, output_file="final_mutation_matrix_for_NN.csv", filter_silent=True):
    """
    Read all MAF files in the specified folder and build a 0/1 mutation feature matrix based on the reference gene list.

    Args:
        maf_folder_path (str): Path to the folder containing MAF files.
        reference_csv_path (str): Path to the reference gene list CSV file exported from R maftools::getGeneSummary().
        output_file (str): Output filename for the final 0/1 matrix.
        filter_silent (bool): Whether to filter out silent mutations.
    """

    # ================= Step 1: Load reference gene list =================
    try:
        print(f"1/5 Loading reference gene list: {reference_csv_path}")
        ref_df = pd.read_csv(reference_csv_path)

        # Ensure 'Hugo_Symbol' exists
        if 'Hugo_Symbol' not in ref_df.columns:
            print("Error: 'Hugo_Symbol' column not found in reference CSV file. Please check R script output.")
            sys.exit(1)

        target_genes = ref_df['Hugo_Symbol'].unique()
        print(f"   -> Reference gene list contains {len(target_genes)} unique genes.")

    except FileNotFoundError:
        print(f"Error: Reference gene list file not found: {reference_csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading reference file: {e}")
        sys.exit(1)

    # ================= Step 2: Read and aggregate all MAF files =================
    print("\n2/5 Reading and aggregating MAF files...")

    all_mutations_list = []
    # Find all .maf.gz files in the folder
    maf_files = glob.glob(os.path.join(maf_folder_path, "*.maf.gz"))

    if not maf_files:
        print(f"Error: No *.maf files found in path {maf_folder_path}.")
        sys.exit(1)
    else:
        print(f"Found {len(maf_files)} files")

    # Define columns to read
    required_cols = ['Hugo_Symbol', 'Tumor_Sample_Barcode']
    if filter_silent:
        required_cols.append('Variant_Classification')

    for file in maf_files:
        try:
            # MAF files are usually Tab-delimited and contain comment lines (starting with #)
            df_temp = pd.read_csv(file, sep='\t', comment='#', usecols=required_cols)
            df_temp['Tumor_Sample_Barcode'] = df_temp['Tumor_Sample_Barcode'].str[:12]
            if filter_silent:
                # Filter out silent mutations
                original_count = len(df_temp)
                df_temp = df_temp[df_temp['Variant_Classification'] != 'Silent']
                filtered_count = len(df_temp)
                # print(f"   -> File {os.path.basename(file)} filtered {original_count - filtered_count} Silent mutations.")

            all_mutations_list.append(df_temp)

        except ValueError as e:
            # Catch errors from usecols, usually due to missing columns in MAF file
            print(f"Warning: File {os.path.basename(file)} is missing required columns. Error: {e}")

        except Exception as e:
            print(f"Failed to read file {file}: {e}")

    # Merge all data
    big_df = pd.concat(all_mutations_list, ignore_index=True)
    if len(big_df) == 0:
        print("Error: No mutation records after aggregation. Please check input files and filter conditions.")
        sys.exit(1)

    print(f"   -> Cumulatively read and retained {len(big_df)} mutation records.")

    # ================= Step 3: Convert to 0/1 matrix (Pivot) =================
    print("\n3/5 Building 0/1 feature matrix...")

    # Use crosstab to count mutations per sample per gene
    mutation_matrix = pd.crosstab(big_df['Tumor_Sample_Barcode'], big_df['Hugo_Symbol'])

    # Binarize: set mutation counts > 0 to 1
    mutation_matrix = (mutation_matrix > 0).astype(int)
    print(f"   -> Initial matrix shape: {mutation_matrix.shape}")

    # ================= Step 4: Align with reference gene list =================
    print("\n4/5 Aligning feature space (Reindex)...")

    # Use reindex to ensure columns match target_genes exactly, missing genes are filled with 0
    final_matrix = mutation_matrix.reindex(columns=target_genes, fill_value=0)

    # Double-check to avoid dimension errors from gene name conflicts
    missing_cols = set(target_genes) - set(final_matrix.columns)
    if missing_cols:
        print(f"Warning: {len(missing_cols)} reference genes never appeared in any sample. Filled with 0.")

    # ================= Step 5: Save results =================
    print(f"\n5/5 Saving results to: {output_file}")
    final_matrix = final_matrix.sort_index(axis=0)
    final_matrix.to_csv(output_file, index=True, compression="gzip")  # Keep Tumor_Sample_Barcode as first column

    print("-" * 50)
    print(f"✅ Processing complete!")
    print(f"Final matrix shape: {final_matrix.shape} (rows: samples, columns: genes)")
    print(f"Output file path: {os.path.abspath(output_file)}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="Convert multiple MAF files into a unified 0/1 mutation feature matrix for neural network models.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '-m', '--maf_dir',
        type=str,
        required=True,
        help="Path to folder containing all MAF files (e.g., ./tcga_mafs)"
    )
    parser.add_argument(
        '-r', '--reference_csv',
        type=str,
        required=True,
        help="Path to CSV file exported from maftools::getGeneSummary() for defining feature space."
    )

    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="final_mutation_matrix_for_NN.csv",
        help="Output filename for the 0/1 feature matrix (default: final_mutation_matrix_for_NN.csv)"
    )
    parser.add_argument(
        '--no_filter_silent',
        action='store_true',
        help="Set this flag to NOT filter silent mutations (default is to filter them)."
    )

    args = parser.parse_args()

    # argparse store_true means if --no_filter_silent is set, its value is True.
    # We need to invert it for the filter_silent parameter.
    filter_silent_mode = not args.no_filter_silent

    print("================== MAF File Matrix Generator ==================")
    print(f"MAF folder: {args.maf_dir}")
    print(f"Reference gene list: {args.reference_csv}")
    print(f"Silent mutation filter: {'Yes (keep non-silent mutations)' if filter_silent_mode else 'No (keep all mutations)'}")
    print("==============================================================\n")

    process_maf_files(
        maf_folder_path=args.maf_dir,
        reference_csv_path=args.reference_csv,
        output_file=args.output,
        filter_silent=filter_silent_mode
    )

if __name__ == "__main__":
    main()