import matplotlib.pyplot as plt

# plt.subplot(121)
# plt.plot(range(100), c='r')
# plt.title('on-off policy')
#
# plt.subplot(122)
# plt.plot(range(100), c='g')
# plt.title('DDQN')
#
# plt.savefig('./logs/image.png')
# plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(range(100), c='r')
ax1.set_title('on-off-policy')
ax2.plot(range(100, 200), c='g')
ax2.set_title('DDQN')
plt.show()