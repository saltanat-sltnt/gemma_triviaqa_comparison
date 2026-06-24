import pandas as pd
import matplotlib.pyplot as plt

# load file
df = pd.read_csv(
    r"C:\Users\Lenovo\Desktop\gemma_triviaqa_comparison\results\api_local_triviaqa\combined_session_summary_20260624_221333.csv")
# sort by number of questions
df = df.sort_values("num_questions")

x = df["num_questions"]
communication = df["communication_time"]
computation = df["computation_time"]

plt.figure(figsize=(8, 5))
plt.bar(x, communication, label="Communication time")
plt.bar(x, computation, bottom=communication, label="Computation time")

plt.xlabel("Number of questions")
plt.ylabel("Time (seconds)")
plt.title("API Time Breakdown by Number of Questions")
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.show()
