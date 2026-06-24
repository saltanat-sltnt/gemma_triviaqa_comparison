import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\Lenovo\Desktop\gemma_triviaqa_comparison\results\api_local_triviaqa\combined_percent_summary_20260624_221333.csv")
# sort by number of questions
df = df.sort_values("num_questions")

x = df["num_questions"]
communication_pct = df["communication_time_percent"]
computation_pct = df["computation_time_percent"]

plt.figure(figsize=(8, 5))
plt.bar(x, communication_pct, label="Communication time %")
plt.bar(x, computation_pct, bottom=communication_pct,
        label="Computation time %")

plt.xlabel("Number of questions")
plt.ylabel("Percentage (%)")
plt.title("Percentage Time Breakdown by Number of Questions")
plt.xticks(x)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()
