import json
import matplotlib.pyplot as plt

# Load benchmark report
with open("reports/robustness_report.json", "r") as f:
    results = json.load(f)

corruptions = []
confidence_drops = []

for result in results:
    corruptions.append(result["corruption"])
    confidence_drops.append(result["confidence_drop"])

# Create graph
plt.figure(figsize=(10, 6))

plt.bar(corruptions, confidence_drops)

plt.xlabel("Corruption Type")

plt.ylabel("Confidence Drop")

plt.title("MRI Robustness Benchmark")

# Save graph
plt.savefig("reports/robustness_plot.png")

print("Robustness visualization saved.")