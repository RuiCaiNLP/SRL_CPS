# Cross-lingual SRL
Codes and data for EMNLP 2020 paper Alignment-free Cross-lingual Semantic Role Labeling. \\
If you have any problem, please email Rui.Cai@ed.ac.uk

## Training ##
- Step1. Preprocess annotated source-language (English) data. One sentence may contain multiple sets of prd-args structure.
        So you should pre-process the data and split them in advance, 
        making sure that one sentence in the data only at maximum contains one set of prd-args proposition.

- Step2. Preprocess parallel data, making sure that they are in the same format with source-language data. 
   - Since parallel data is unlabled, you need to perform predicate identification and find predicate pairs in advance.
   - All data after prepocessing should be placed under directory ./temp


- Step3. Start the cross-lingual training.
   - sh train.sh
         
## Manually Annotated Data ##
- We manuanlly annotated 304 Chinese samples and 258 German Samples for evaluation: [New datasets available here](https://github.com/RuiCaiNLP/ZH_DE_Datasets)
