# SRL_CPS
codes and data for EMNLP 2020 paper Alignment-free Cross-lingual Semantic Role Labeling

## Training ##
- Step1. Preprocess annotated source-language (English) data. One sentence may contain multiple sets of prd-args structure.
        So you should pre-process the data and split them in advance, 
        making sure that one sentence in the data only at maximum contains one set of prd-args proposition.

- Step2. Preprocess parallel data, making sure that they are in the same format with source-language data. All data after prepocessing
         should be placed under directory ./temp


- Step3. Start the cross-lingual training.
   - sh train.sh
         

