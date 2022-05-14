import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


r"""lr vs coeff"""
plt.xlabel("lr")
plt.ylabel("coeff")
x = [1, 2, 3, 4, 5, 6, 7, 8]
x_index = ['0.00005', '0.0001', '0.0002', '0.0005', '0.001', '0.005', '0.01', '0.02']
# std V: [0.028, 0.008, 0.005, 0.013, 0.012, 0.005, 0.005, 0.010]
Vcoeff = [0.853, 0.894, 0.905, 0.881, 0.886, 0.892, 0.912, 0.780]
# std Q: [0.002, 0.001, 0.000, 0.000, 0.000, 0.001, 0.005, 0.003]
Qcoeff = [0.987, 0.992, 0.995, 0.996, 0.996, 0.995, 0.991, 0.970]
y_locator = MultipleLocator(0.02)
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
plt.ylim(0.770, 1.01)
_ = plt.xticks(x, x_index)
plt.xlim(0.5, 8.5)
plt.title("lr vs coeff")
plt.savefig('D:/ERG4901/ViT_Project/lr_coeff.eps')
plt.show()


r"""lr vs loss"""
plt.xlabel("lr")
plt.ylabel("loss")
x = [1, 2, 3, 4, 5, 6, 7, 8]
x_index = ['0.00005', '0.0001', '0.0002', '0.0005', '0.001', '0.005', '0.01', '0.02']
# std V:[0.000352, 0.000058, 0.000079, 0.000164, 0.000225, 0.000208, 0.000066, 0.000158]
Vloss = [0.001856, 0.001305, 0.001210, 0.001546, 0.001358, 0.001512, 0.001087, 0.002606]
# std Q:[0.000061, 0.000007, 0.000007, 0.000044, 0.000009, 0.000036, 0.000111, 0.000030]
Qloss = [0.000319, 0.000191, 0.000122, 0.000120, 0.000096, 0.000137, 0.000215, 0.000745]
# y_locator = MultipleLocator(0.0001)
# ax = plt.gca()
# ax.yaxis.set_major_locator(y_locator)
V, = plt.semilogy(x, Vloss, linewidth=3.0)
Q, = plt.semilogy(x, Qloss, linewidth=3.0)
# for a, b in zip(x, Vloss):
    # plt.text(a, b, "{:.6f}".format(b), va='bottom', ha='left', rotation=45)
#     plt.text(a, b, "{:.6f}".format(b), va='bottom', ha='left')
# for c, d in zip(x, Qloss):
#     plt.text(c, d, "{:.6f}".format(d), va='top', ha='left')
plt.legend([V, Q], ["Vloss", "Qloss"])
plt.ylim(0.00005, 0.005)
_ = plt.xticks(x, x_index)
plt.xlim(0.5, 8.5)
plt.title("lr vs loss")
plt.savefig('D:/ERG4901/ViT_Project/lr_loss.eps')
plt.show()


r"""lr vs error"""
plt.xlabel("lr")
plt.ylabel("pred error")
x = [1, 2, 3, 4, 5, 6, 7, 8]
x_index = ['0.00005', '0.0001', '0.0002', '0.0005', '0.001', '0.005', '0.01', '0.02']
# std V min= [0.00890705, 0.00070890, 0.00175485, 0.00650096, 0.00074568, 0.00418190, 0.00048273, 0.00651197]
Verror_min = [0.00536764, 0.00052305, 0.00128943, 0.00546875, 0.00063197, 0.00363448, 0.00087994, 0.00499915]
# std Q min= [0.00007689, 0.00005676, 0.00004088, 0.00002782, 0.00028033, 0.00008664, 0.00005216, 0.00099129]
Qerror_min = [0.00006056, 0.00005193, 0.00007496, 0.00003749, 0.00017883, 0.00008359, 0.00005767, 0.00061125]
# std V conv= [0.11417050, 0.03935889, 0.03656990, 0.06274128, 0.03150194, 0.06943224, 0.06234056, 0.12320508]
Verror_conv = [0.11679122, 0.04266955, 0.06141586, 0.19592018, 0.03685992, 0.13173735, 0.08568188, 0.13981960]
# std Q conv= [0.00453396, 0.00444021, 0.00242363, 0.00143254, 0.00167777, 0.00249757, 0.00121917, 0.00621423]
Qerror_conv = [0.00607648, 0.00661471, 0.00511929, 0.00233044, 0.00537258, 0.00394127, 0.00521658, 0.01366351]
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
plt.ylim(0.000001, 0.8)
_ = plt.xticks(x, x_index)
plt.xlim(0.5, 8.5)
plt.title("lr vs error")
plt.savefig('D:/ERG4901/ViT_Project/lr_error.eps')
plt.show()


r"""ABS vs GELU"""
plt.xlabel("lr")
plt.ylabel("V coeff")
x = [1, 2, 3, 4, 5, 6, 7, 8]
x_index = ['0.00005', '0.0001', '0.0002', '0.0005', '0.001', '0.005', '0.01', '0.02']
# std ABS =  [0.028, 0.008, 0.005, 0.013, 0.012, 0.005, 0.005, 0.010]
Vcoeff_ABS = [0.853, 0.894, 0.905, 0.881, 0.886, 0.892, 0.912, 0.780]
# std GELU =  [0.003, 0.010, 0.010, 0.013, 0.012, 0.005, 0.018, 0.009]
Vcoeff_GELU = [0.799, 0.826, 0.869, 0.846, 0.877, 0.890, 0.908, 0.758]
y_locator = MultipleLocator(0.01)
ax = plt.gca()
ax.yaxis.set_major_locator(y_locator)
V_ABS, = plt.plot(x, Vcoeff_ABS, linewidth=3.0)
V_GELU, = plt.plot(x, Vcoeff_GELU, linewidth=3.0)
# ax.semilogy(x, Verror_min)
# ax.semilogy(x, Qerror_min)
# ax.semilogy(x, Verror_conv)
# for a, b in zip(x, Vloss):
    # plt.text(a, b, "{:.6f}".format(b), va='bottom', ha='left', rotation=45)
#     plt.text(a, b, "{:.6f}".format(b), va='bottom', ha='left')
# for c, d in zip(x, Qloss):
#     plt.text(c, d, "{:.6f}".format(d), va='top', ha='left')
plt.legend([V_ABS, V_GELU], ["ABS", "GELU"])
plt.ylim(0.75, 0.95)
_ = plt.xticks(x, x_index)
plt.xlim(0.5, 8.5)
plt.title("act vs coeff")
plt.savefig('D:/ERG4901/ViT_Project/act_coeff.eps')
plt.show()

