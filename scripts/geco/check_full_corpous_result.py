import pandas as pd
df = pd.read_csv("data/geco/benchmark/full_corpus_results.csv")
# 只看包含 Acc, Top3, Rec 的平均值
print(df.filter(regex='Acc|Top3|Rec').mean().round(2))