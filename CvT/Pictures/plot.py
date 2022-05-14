import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


r"""lr vs coeff"""
plt.xlabel("lr")
plt.ylabel("coeff")
x = [1, 2, 3, 4, 5]
x_index = ['0.0001', '0.00025', '0.001', '0.0025', '0.01']
Vcoeff = [0.805, 0.853, 0.888, 0.876, 0.886]
Qcoeff = [0.987, 0.990, 0.994, 0.993, 0.993]
y_locator = MultipleLocator(0.05)
ax = plt.gca()
ax.yaxis.set_major_locator(y_locator)
V, = plt.plot(x, Vcoeff, linewidth=3.0)
Q, = plt.plot(x, Qcoeff, linewidth=3.0)
# for a, b in zip(x, Vcoeff):
    # plt.text(a, b, "{:.3f}".format(b), va='bottom', ha='left', rotation=45)
    # plt.text(a, b, "{:.3f}".format(b), va='bottom', ha='left')
# for c, d in zip(x, Qcoeff):
#     plt.text(c, d, "{:.3f}".format(d), va='top', ha='left')
plt.legend([V, Q], ["Vcoeff", "Qcoeff"])
plt.ylim(0.750, 1.010)
_ = plt.xticks(x, x_index)
plt.xlim(0.5, 5.5)
plt.title("lr vs coeff")
plt.savefig('D:/ERG4901/CvT_Project/lr_coeff.eps')
plt.show()


r"""lr vs loss"""
plt.xlabel("lr")
plt.ylabel("loss")
x = [1, 2, 3, 4, 5]
x_index = ['0.0001', '0.00025', '0.001', '0.0025', '0.01']
Vloss = [0.002398, 0.001720, 0.001458, 0.001531, 0.001338]
Qloss = [0.000329, 0.000245, 0.000140, 0.000180, 0.000181]
y_locator = MultipleLocator(0.0005)
ax = plt.gca()
ax.yaxis.set_major_locator(y_locator)
V, = plt.plot(x, Vloss, linewidth=3.0)
Q, = plt.plot(x, Qloss, linewidth=3.0)
# for a, b in zip(x, Vloss):
    # plt.text(a, b, "{:.6f}".format(b), va='bottom', ha='left', rotation=45)
#     plt.text(a, b, "{:.6f}".format(b), va='bottom', ha='left')
# for c, d in zip(x, Qloss):
#     plt.text(c, d, "{:.6f}".format(d), va='top', ha='left')
plt.legend([V, Q], ["Vloss", "Qloss"])
plt.ylim(0, 0.0027)
_ = plt.xticks(x, x_index)
plt.xlim(0.5, 5.5)
plt.title("lr vs loss")
plt.savefig('D:/ERG4901/CvT_Project/lr_loss.eps')
plt.show()


r"""lr vs error"""
plt.xlabel("lr")
plt.ylabel("pred error")
x = [1, 2, 3, 4, 5]
x_index = ['0.0001', '0.00025', '0.001', '0.0025', '0.01']
Verror_min = [0.000108, 0.001754, 0.002630, 0.001380, 0.000491]
Qerror_min = [0.000101, 0.000065, 0.000106, 0.000061, 0.000008]
Verror_conv = [0.1120, 0.0603, 0.1992, 0.0411, 0.0724]
Qerror_conv = [0.00813, 0.00842, 0.00975, 0.00895, 0.03130]
# y_locator = MultipleLocator(0.0005)
ax = plt.gca()
# ax.yaxis.set_major_locator(y_locator)
V_min, = plt.semilogy(x, Verror_min, linewidth=3.0, color='blue', linestyle='solid')
Q_min, = plt.semilogy(x, Qerror_min, linewidth=3.0, color='red', linestyle='solid')
V_conv, = plt.semilogy(x, Verror_conv, linewidth=3.0, color='blue', linestyle='dashed')
Q_conv, = plt.semilogy(x, Qerror_conv, linewidth=3.0, color='red', linestyle='dashed')
# ax.semilogy(x, Verror_min)
# ax.semilogy(x, Qerror_min)
# ax.semilogy(x, Verror_conv)
# for a, b in zip(x, Vloss):
    # plt.text(a, b, "{:.6f}".format(b), va='bottom', ha='left', rotation=45)
#     plt.text(a, b, "{:.6f}".format(b), va='bottom', ha='left')
# for c, d in zip(x, Qloss):
#     plt.text(c, d, "{:.6f}".format(d), va='top', ha='left')
plt.legend([V_min, Q_min, V_conv, Q_conv], ["min V error", "min Q error", "conv V error", "conv Q error"])
plt.ylim(0.000001, 0.3)
_ = plt.xticks(x, x_index)
plt.xlim(0.5, 5.5)
plt.title("lr vs error")
plt.savefig('D:/ERG4901/CvT_Project/lr_error.eps')
plt.show()



