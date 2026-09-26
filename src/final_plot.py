import matplotlib.pyplot as plt

labels = ["Embed", "ResidPre\nL0", "ResidMid\nL0", "MLP\nL0", "ResidPost\nL6", "ResidPost\nL8", "ResidPost\nL10"]
paris_london = [100.0, 100.0, 100.0, 74.7, 100.0, 100.0, 100.0]
ioi = [0.0, 0.0, -0.1, -0.1, 1.5, 101.2, 102.8]

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(labels, paris_london, marker="o", linewidth=2.5, label="Fact recall (Paris/London)", color="#4C72B0")
ax.plot(labels, ioi, marker="o", linewidth=2.5, label="Entity tracking (IOI)", color="#DD8452")

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_ylabel("% of behavior recovered", fontsize=12)
ax.set_title("Where in the network is the behavior actually computed?", fontsize=14, weight="bold", pad=15)
ax.legend(fontsize=11, loc="center right")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-15, 115)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("recovery_comparison.png", dpi=200, bbox_inches="tight")
plt.show()