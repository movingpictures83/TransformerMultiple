# TransformerMultiple
# Language: Python
# Input: TXT
# Output: PNG
# Tested with: PluMA 1.1, Python 3.6

PluMA plugin that runs Transformer model (Vaswani et al, 2017)

The plugin expectes as input a tab-delimited file of keyword-value pairs:
inputfile: Dataset
divide: Row where dataset starts
lr: Learning RAte
epochs: Number of epochs
target: Target column
stations: Columns to plot
indices: Indices of stations

The graph will be output in PNG format
