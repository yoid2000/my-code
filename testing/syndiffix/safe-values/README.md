
Put data_dense.csv and data_sparse.csv into directory salary_data2.

Run syn.py (from the parent directory of salary_data2) to generate the needed synthetic files as parquet.

Run pdfs2.py to generate the current plots for the paper.

The files needed by pdfs2.py are in salary_data2 (ending with either .csv or .parquet).

pdfs2.py generates the .png, .pdf, and .tex files in salary_data2.