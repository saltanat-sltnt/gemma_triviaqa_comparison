import pandas as pd

df = pd.read_csv(
    r"C:\Users\Lenovo\Desktop\gemma_triviaqa_comparison\results\api_local_triviaqa\combined_percent_summary_20260624_223132.csv"
)

table = df[
    [
        "num_questions",
        "seed",

        "send_request_time_percent",
        "network_latency_percent",
        "receive_response_time_percent",

        "processing_non_inference_time_percent",
        "inference_time_percent",

        "communication_time_percent",
        "computation_time_percent",
    ]
].copy()

table = table.rename(
    columns={
        "num_questions": "Questions",
        "seed": "Seed",

        "send_request_time_percent": "Send request / client total (%)",
        "network_latency_percent": "Network latency / client total (%)",
        "receive_response_time_percent": "Receive response / client total (%)",

        "processing_non_inference_time_percent": "Processing / cloud total (%)",
        "inference_time_percent": "Inference / cloud total (%)",

        "communication_time_percent": "Communication / API total (%)",
        "computation_time_percent": "Computation / API total (%)",
    }
)

# Create average row safely
avg_row = {
    "Questions": "Average",
    "Seed": "-",
}

for col in table.columns:
    if col not in ["Questions", "Seed"]:
        avg_row[col] = table[col].mean()

table = pd.concat([table, pd.DataFrame([avg_row])], ignore_index=True)

# Round numeric columns
for col in table.columns:
    if col not in ["Questions", "Seed"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(2)

output_path = r"C:\Users\Lenovo\Desktop\gemma_triviaqa_comparison\results\api_local_triviaqa\percentage_comparison_table.csv"
table.to_csv(output_path, index=False)

print(table)
print(f"\nSaved to: {output_path}")
