import os
import pandas as pd

# Input and output paths
input_files = {
    "data/Human__TCGA_BRCA__BDGSC__miRNASeq__HS_miR__01_28_2016__BI__Gene__Firehose_RPKM_log2.cct": "data/gene_expression.csv",
    "data/Human__TCGA_BRCA__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct": "data/dna_methylation.csv",
    "data/Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct": "data/mirna_expression.csv",
    "data/Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi": "data/survival.csv"
}

# ID normalization function
def normalize_id(s):
    return s.strip().lower().replace('"', '').replace('.', '-')

for infile, outfile in input_files.items():
    try:
        # Read the file (tab-separated, first column as index)
        df = pd.read_csv(infile, sep="\t", index_col=0)

        # Transpose omics data so samples are rows, if not clinical
        if infile.endswith(".cct"):
            df = df.transpose()

        # Normalize sample IDs (index and/or columns)
        df.index = [normalize_id(i) for i in df.index]
        df.columns = [normalize_id(c) for c in df.columns]

        # Save as CSV
        df.to_csv(outfile)
        print(f"✅ Converted '{infile}' → '{outfile}'")

    except Exception as e:
        print(f"❌ Error converting '{infile}': {e}")
