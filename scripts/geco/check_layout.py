import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/geco/geco_pp01_cognitive_mass.csv')
plt.figure(figsize=(10, 6))
plt.scatter(df['true_x'], df['true_y'], alpha=0.5)
for i, txt in enumerate(df['WORD']):
    plt.annotate(txt, (df['true_x'].iloc[i], df['true_y'].iloc[i]), fontsize=8)
plt.gca().invert_yaxis()
plt.title('GECO pp01 Trial 5 Word Positions')
plt.xlabel('true_x')
plt.ylabel('true_y')
plt.savefig('docs/figures/geco_data_layout.png')
print("✅ Layout plot saved to docs/figures/geco_data_layout.png")
