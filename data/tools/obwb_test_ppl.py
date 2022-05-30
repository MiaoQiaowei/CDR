import numpy as np
import matplotlib.pyplot as plt

x = np.array([30.002,
49.511,
62.289,
70.968,
76.178,
79.224,
81.282,
82.394,
83.284])
y = np.array([12.249,
11.55,
11.015,
10.581,
10.212,
9.991,
9.687,
9.505,
9.345])

x1 = np.array([17.511,
33.729,
41.899,
49.05,
53.16,
54.98,
55.90,
76])
y1 = np.array([10.2574,
10.158,
10.05,
9.92,
9.82,
9.7,
9.7,
8.3])

fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(x, y, dashes=[2, 2], label='MIC', color="r")


# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax.plot(x1, y1, dashes=[6, 2], label='ComiRec-SA', color="k")
ax.set_xlim([13, 80])
ax.set_ylim([8, 13])
ax.legend()
# plt.show()
plt.savefig("./diversity.pdf", format="pdf")
print("test")