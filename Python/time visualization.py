#draws plot visualizing how time depends on image size
import matplotlib.pyplot as plt

a = [100, 200, 300, 400, 500]
b1 = [0.26495981216430664, 0.9432852268218994, 2.190534830093384, 3.8519835472106934, 6.019513845443726]
b2 = [4.909441947937012, 42.037320375442505, 141.71538305282593, 343.69941997528076, 674.3807723522186]
b3 = [0.5788874626159668, 3.470895528793335, 9.808711051940918, 18.64536213874817, 29.94342565536499]


plt.xlabel("size of image (one side)")
plt.ylabel("time in seconds")
plt.plot(a, b1, label = "Dijkstra")
plt.plot(a, b2, label = "Naive")
plt.plot(a, b3, label = "Neighbourhood")
plt.legend()
plt.show()