import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path(r"results/local/combined_session_summary_20260630_110424.csv")

df = pd.read_csv(csv_path)
df = df.sort_values("num_questions")

x_labels = df["num_questions"].astype(str)
processing_time = df["processing_time"]
inference_time = df["inference_time"]

plt.figure(figsize=(8, 5))

plt.bar(x_labels, processing_time, label="Processing time")
plt.bar(x_labels, inference_time, bottom=processing_time, label="Inference time")

plt.xlabel("Number of questions")
plt.ylabel("Time (seconds)")
plt.title("Local time breakdown by number of questions")
plt.legend()
plt.tight_layout()

output_path = csv_path.parent / "local_time_graph_20260630_110424.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved graph to: {output_path}")
