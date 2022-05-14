import torchvision.transforms as transforms
import h5py
import numpy as np
import torch
import torch.nn as nn
import vision_transformer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import torch.optim as optim

n_epochs = 300
log_interval = 10
batch_size_train = 64
batch_size_test = 100
alpha = 5
device = torch.device('cuda')
ModelPATH_bestVloss = 'D:\ERG4901\ViT_Project\model_bestV.pt'
ModelPATH_bestQloss = 'D:\ERG4901\ViT_Project\model_bestQ.pt'


class TensorsDataset(torch.utils.data.Dataset):
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''

    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor

    def __len__(self):
        return self.data_tensor.size(0)


f = h5py.File('D:/ERG4901/ViT_Project/Input.mat', 'r')
data = f['Input']
Input = np.array(data)  # For converting to a NumPy array

f = h5py.File('D:/ERG4901/ViT_Project/Output.mat', 'r')
data = f['QnV']
Output = np.array(data)  # For converting to a NumPy array

# f = h5py.File('D:/ERG4901/ViT_Project/fixed_train_loss/TestXdata.mat', 'r')
# data = f['xtest']
# Input = np.array(data)
#
# f = h5py.File('D:/ERG4901/ViT_Project/fixed_train_loss/TestYdata.mat', 'r')
# data = f['ytest']
# Output = np.array(data)


"""converting Input to tensor with shape [12500, 3, 5, 12]"""
input_tensor = torch.tensor(Input).permute(3, 2, 1, 0).float()
print("input tensor shape: ", input_tensor.shape)
# print("input_tensor example: ", input_tensor[0])
"""converting QnV to tensor with shape [12500, 2]"""
output_tensor = torch.tensor(Output).permute(1, 0).float()
output_tensor = output_tensor.view(-1, 2)  # correct the dimension
# transformer = transforms.Compose([transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
#                                   transforms.Normalize(mean=[-8.7270e-13, 3.3969e-13, -1.6978e-12],
#                                                        std=[0.0000000005, 0.0000000005, 0.0000000005])
#                                   ])
transformer = transforms.Normalize(mean=[-8.7270e-13, 3.3969e-13, -1.6978e-12],
                                   std=[0.0000000005, 0.0000000005, 0.0000000005])
dataset = TensorsDataset(input_tensor, output_tensor, transforms=transformer)

# split into training and test datasets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
# load the data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)


r"""ViT有个act layer注意一下"""
def ViT1():
    # 中
    return vision_transformer.ViT(image_size=(6, 12), num_classes=2, mlp_dim=72, depth=6, patch_size=2, pool='cls',
                                  channels=3, dim=36, heads=6, dim_head=12, dropout=0., emb_dropout=0.)


def ViT2():
    # 大
    return vision_transformer.ViT(image_size=(6, 12), num_classes=2, mlp_dim=144, depth=6, patch_size=2, pool='cls',
                                  channels=3, dim=72, heads=12, dim_head=24, dropout=0., emb_dropout=0.)


def ViT3():
    # 大
    return vision_transformer.ViT(image_size=(6, 12), num_classes=2, mlp_dim=144, depth=6, patch_size=2, pool='cls',
                                  channels=3, dim=24, heads=3, dim_head=8, dropout=0., emb_dropout=0.)


r"""注意plot那边也要把模型换了(两处)"""
start = time.time()
network = ViT1().to(device)
# optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

############try AdamW
optimizer = optim.Adam(network.parameters(), lr=0.0002)
# optimizer = optim.AdamW(network.parameters(), lr=0.0002)
# optimizer = optim.ASGD(network.parameters(), lr=0.0001, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 160, 200], gamma=0.1)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel')
# lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
# lr_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

train_counter = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

trainQ_losses = []  # for Q
trainV_losses = []  # for V
testQ_losses = []  # for Q
testV_losses = []  # for V

trainQ_output = []  # for Q
trainV_output = []  # for V
trainQ_target = []  # for Q
trainV_target = []  # for V

testQ_output = []  # for Q
testV_output = []  # for V
testQ_target = []  # for Q
testV_target = []  # for V


def train(epoch,alpha,train_loader):
    # lr_scheduler should be used after optimizer
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = network(data)
        # Save the training result
        # Q的预测输出
        trainQ_output.append(output.data[:, 0])
        trainQ_target.append(target.data[:, 0])
        # V的预测输出
        trainV_output.append(output.data[:, 1])
        trainV_target.append(target.data[:, 1])
        Q_loss = F.mse_loss(output[:, 0], target[:, 0])
        V_loss = F.mse_loss(output[:, 1], target[:, 1])
        loss = Q_loss + alpha * V_loss
        ############
        # calculate the gradient
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)   # 不好用就删掉
        optimizer.step()
        # gradually print epoch results
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tQloss:{:.6f}\tVloss:{:.6f}\tQ_NN: {:.4f}\tV_NN: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                Q_loss.item(), V_loss.item(), output[-1, 0], output[-1, 1]))
            # store training results
            trainQ_losses.append(Q_loss.item())
            trainV_losses.append(V_loss.item())
            train_counter.append((batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))

            for param_group in optimizer.param_groups:
                print("Current learning rate: {}".format(param_group['lr']))
    lr_scheduler.step()
    # lr_scheduler2.step()


def test():
    # global test_output
    network.eval()
    testQ_loss = 0
    testV_loss = 0
    with torch.no_grad():  # disable the gradient computation
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = network(data)
            # save the test result
            # Q
            testQ_output.append(output[:, 0])
            testQ_target.append(target[:, 0])
            # V
            testV_output.append(output[:, 1])
            testV_target.append(target[:, 1])
            # loss
            testQ_loss += F.mse_loss(output[:, 0], target[:, 0]).item()
            testV_loss += F.mse_loss(output[:, 1], target[:, 1]).item()

    # calculate the average loss per epoch
    testQ_loss /= 25
    testV_loss /= 25
    testQ_losses.append(testQ_loss)
    testV_losses.append(testV_loss)
    print('\nTest set: Qloss: {:.6f}, Vloss: {:.6f}, Qn: {:.4f}, Qf: {:.4f}, Vn: {:.4f}, Vf: {:.4f}\n'.format(
        testQ_loss, testV_loss, output[-1, 0], target[-1, 0], output[-1, 1], target[-1, 1]))

    return testQ_loss, testV_loss


# run the training
print("start training")

test()
best_testlossQ = 100000.
best_testlossV = 100000.
for epoch in range(1, n_epochs + 1):
    train(epoch,alpha,train_loader)
    QL, VL = test()
    if best_testlossQ > QL:
        best_testlossQ = QL
        torch.save(network.state_dict(), ModelPATH_bestQloss)
    if best_testlossV > VL:
        best_testlossV = VL
        torch.save(network.state_dict(), ModelPATH_bestVloss)

print("finish training")
end = time.time()
print("time cost: %.2f min" % ((end - start) / 60.0))

print("-------------------------Summary--------------------------")


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""

    # Compute correlation matrix
    corr_mat = np.corrcoef(x, y)

    # calculate the coeff
    pearson_R = corr_mat[0, 1]

    # Return entry [0,1]
    return pearson_R


def pred_error(x, y):
    """Compute absolute percentage error between two arrays."""

    diff = np.absolute(x - y)
    percent_e = 100 * diff / y
    return percent_e


# Post-processing of data

print(len(trainQ_output))
print(len(testQ_output))

print(len(trainV_output))
print(len(testV_output))

# obtain and convert learning results from list to tensor
trainQ_output = torch.cat(trainQ_output, 0)
trainQ_target = torch.cat(trainQ_target, 0)

trainV_output = torch.cat(trainV_output, 0)
trainV_target = torch.cat(trainV_target, 0)

testQ_output = torch.cat(testQ_output, 0)
testQ_target = torch.cat(testQ_target, 0)

testV_output = torch.cat(testV_output, 0)
testV_target = torch.cat(testV_target, 0)

print(len(trainQ_target))
print(len(testQ_target))

trainQ_outputArr = trainQ_output.cpu().numpy()
trainQ_targetArr = trainQ_target.cpu().numpy()

trainV_outputArr = trainV_output.cpu().numpy()
trainV_targetArr = trainV_target.cpu().numpy()

testQ_outputArr = testQ_output.cpu().numpy()
testQ_targetArr = testQ_target.cpu().numpy()

testV_outputArr = testV_output.cpu().numpy()
testV_targetArr = testV_target.cpu().numpy()

# print('Training dataset size: {}'.format(len(trainV_outputArr)))
# print('Test dataset size: {}'.format(len(testV_outputArr)))

# print("train_outputArr: ", len(trainQ_outputArr))

predError_train = []
predError_test = []
predErrorV_train = []
predErrorV_test = []

for i in range(n_epochs):
    predError_train.append(pred_error(np.mean(trainQ_outputArr[i * 10000:(i + 1) * 10000 - 1]),
                                      np.mean(trainQ_targetArr[i * 10000:(i + 1) * 10000 - 1])))
    predErrorV_train.append(pred_error(np.mean(trainV_outputArr[i * 10000:(i + 1) * 10000 - 1]),
                                       np.mean(trainV_targetArr[i * 10000:(i + 1) * 10000 - 1])))

    t = i + 1
    predError_test.append(pred_error(np.mean(testQ_outputArr[t * 2500:(t + 1) * 2500 - 1]),
                                     np.mean(testQ_targetArr[t * 2500:(t + 1) * 2500 - 1])))
    predErrorV_test.append(pred_error(np.mean(testV_outputArr[t * 2500:(t + 1) * 2500 - 1]),
                                      np.mean(testV_targetArr[t * 2500:(t + 1) * 2500 - 1])))

print('Q Min pred error train: {:.8f}\nQ Min pred error test: {:.8f}'.format(min(predError_train), min(predError_test)))
print('Q Conv pred error train: {:.8f}\nQ Conv pred error test: {:.8f}'.format(np.mean(predError_train[len(predError_train)-10:len(predError_train)-1]),
                                                                               np.mean(predError_test[len(predError_test)-10:len(predError_test)-1])))
# print('Min mse train: {:.8f}\nMin mse test: {:.8f}'.format(min(trainQ_losses), min(testQ_losses)))
#
print('V Min pred error train: {:.8f}\nV Min pred error test: {:.8f}'.format(min(predErrorV_train), min(predErrorV_test)))
print('V Conv pred error train: {:.8f}\nV Conv pred error test: {:.8f}'.format(np.mean(predErrorV_train[len(predErrorV_train)-10:len(predErrorV_train)-1]),
                                                                               np.mean(predErrorV_test[len(predErrorV_test)-10:len(predErrorV_test)-1])))
# print('V Min mse train: {:.8f}\nV Min mse test: {:.8f}'.format(min(trainV_losses), min(testV_losses)))

# %% plots
"""Q Loss"""
fig = plt.figure()
plt.plot([x / 10000 for x in train_counter], trainQ_losses, color='blue')
plt.plot([x / 10000 for x in test_counter], testQ_losses, linewidth=2, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Q Mean Squared Error')
# plt.ylim(-0.015,0.25)
plt.yscale('log')
plt.savefig('D:/ERG4901/ViT_Project/Q_MSE_loss.eps')
plt.show()

"""V Loss"""
fig = plt.figure()
plt.plot([x / 10000 for x in train_counter], trainV_losses, color='blue')
plt.plot([x / 10000 for x in test_counter], testV_losses, linewidth=2, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('V Mean Squared Error')
# plt.ylim(-0.015,0.25)
plt.yscale('log')
plt.savefig('D:/ERG4901/ViT_Project/V_MSE_loss.eps')
plt.show()

"""Q prediction error"""
fig = plt.figure()
plt.plot(np.linspace(1, n_epochs, num=n_epochs), predError_train, color='blue')
plt.plot(np.linspace(1, n_epochs, num=n_epochs), predError_test, color='red')
plt.legend(['Train error', 'Test error'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Prediction Error of Q (%)')
plt.yscale('log')
plt.savefig('D:/ERG4901/ViT_Project/Q_pred_error.eps')
plt.show()

"""V prediction error"""
fig = plt.figure()
plt.plot(np.linspace(1, n_epochs, num=n_epochs), predErrorV_train, color='blue')
plt.plot(np.linspace(1, n_epochs, num=n_epochs), predErrorV_test, color='red')
plt.legend(['Train error', 'Test error'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Prediction Error of V (%)')
plt.yscale('log')
plt.savefig('D:/ERG4901/ViT_Project/V_pred_error.eps')
plt.show()


def test_epoch(model, loader):
    network.eval()
    Q_loss = 0
    V_loss = 0
    Q_out =[]
    V_out =[]
    Q_tarout = []
    V_tarout = []

    with torch.no_grad():  # disable the gradient computation
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            # save the test result
            # Q
            Q_out.append(output[:, 0])
            Q_tarout.append(target[:, 0])
            # V
            V_out.append(output[:, 1])
            V_tarout.append(target[:, 1])
            # loss
            Q_loss += F.mse_loss(output[:, 0], target[:, 0]).item()
            V_loss += F.mse_loss(output[:, 1], target[:, 1]).item()

    # calculate the average loss per epoch
    Q_loss /= 25
    V_loss /= 25
    testQ_losses.append(Q_loss)
    testV_losses.append(V_loss)
    print('\nTest set: Qloss: {:.6f}, Vloss: {:.6f}\n'.format(Q_loss, V_loss))

    return Q_loss, V_loss, Q_out, Q_tarout, V_out, V_tarout


# Q loss最好的模型， V loss不一定好
modelQ = ViT1().to(device)
modelQ.load_state_dict(torch.load(ModelPATH_bestQloss))

trainQ_loss, trainV_loss, Q_out, Q_tarout, V_out, V_tarout = test_epoch(modelQ, train_loader)
print('\nBest Q train: Qloss: {:.6f}\n'.format(trainQ_loss))

Q_out = torch.cat(Q_out, 0)
Q_tarout = torch.cat(Q_tarout, 0)
V_out = torch.cat(V_out, 0)
V_tarout = torch.cat(V_tarout, 0)

Q_outArr = Q_out.detach().cpu().numpy()
Q_taroutArr = Q_tarout.detach().cpu().numpy()
V_outArr = V_out.cpu().detach().numpy()
V_taroutArr = V_tarout.detach().cpu().numpy()

coeffQ_train_best = pearson_r(Q_outArr, Q_taroutArr)
print('Best test Q corr coeff: {:.3f}'.format(coeffQ_train_best))
coeffQ_train_best = round(coeffQ_train_best, 3)

"""Q coefficient train"""
fig = plt.figure()

plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.scatter(10 ** Q_taroutArr, 10 ** Q_outArr, s=5, color='purple')
plt.legend(['Training data (10k)\ncorrelation coeff ' + "{:.3f}".format(coeffQ_train_best)], loc='upper right')
x = np.linspace(0, 8e5, 10)
y = x
plt.plot(x, y, c='k')
plt.xlabel('Q_FDTD')
plt.ylabel('Q_NN')
plt.xlim(1e+5, 7e+5)
plt.ylim(1e+5, 7e+5)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('D:/ERG4901/ViT_Project/Qcorr_coeff_train.eps')
plt.show()


r""""""
testQ_loss, testV_loss, Q_out, Q_tarout, V_out, V_tarout = test_epoch(modelQ, test_loader)
print('\nBest Q: Qloss: {:.6f}\n'.format(testQ_loss))

Q_out = torch.cat(Q_out, 0)
Q_tarout = torch.cat(Q_tarout, 0)
V_out = torch.cat(V_out, 0)
V_tarout = torch.cat(V_tarout, 0)

Q_outArr = Q_out.cpu().numpy()
Q_taroutArr = Q_tarout.cpu().numpy()
V_outArr = V_out.cpu().numpy()
V_taroutArr = V_tarout.cpu().numpy()

coeffQ_test_best = pearson_r(Q_outArr, Q_taroutArr)
print('Best test Q corr coeff: {:.3f}'.format(coeffQ_test_best))
coeffQ_test_best = round(coeffQ_test_best, 3)


"""Q coefficient test"""
fig = plt.figure()
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.scatter(10 ** Q_taroutArr, 10 ** Q_outArr, s=5, color='purple')
plt.legend(['Test data (2500)\ncorrelation coeff ' + "{:.3f}".format(coeffQ_test_best)], loc='upper right')
x = np.linspace(0, 8e5, 10)
y = x
plt.plot(x, y, c='k')
plt.xlabel('Q_FDTD')
plt.ylabel('Q_NN')
plt.axis('square')
plt.xlim(1e+5, 7e+5)
plt.ylim(1e+5, 7e+5)
plt.savefig('D:/ERG4901/ViT_Project/Qcorr_coeff_test.eps')
plt.show()



# V loss最好的模型， Q loss不一定好
modelV = ViT1().to(device)
modelV.load_state_dict(torch.load(ModelPATH_bestVloss))

trainQ_loss, trainV_loss, Q_out, Q_tarout, V_out, V_tarout = test_epoch(modelV, train_loader)
print('\nBest V train: Vloss: {:.6f}\n'.format(trainV_loss))

Q_out = torch.cat(Q_out, 0)
Q_tarout = torch.cat(Q_tarout, 0)
V_out = torch.cat(V_out, 0)
V_tarout = torch.cat(V_tarout, 0)
Q_outArr = Q_out.cpu().numpy()  # for Q
Q_taroutArr = Q_tarout.cpu().numpy()
V_outArr = V_out.cpu().numpy()  # for V
V_taroutArr = V_tarout.cpu().numpy()

coeffV_train_best = pearson_r(V_outArr, V_taroutArr)
print('Best train V corr coeff: {:.3f}'.format(coeffV_train_best))
coeffV_train_best = round(coeffV_train_best, 3)

"""V coefficient train"""
fig = plt.figure()
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.scatter(V_taroutArr, V_outArr, s=5, color='purple')
plt.legend(['Training data (10k)\ncorrelation coeff ' + "{:.3f}".format(coeffV_train_best)], loc='upper right')
x = np.linspace(0, 1.5, 5)
y = x
plt.plot(x, y, c='k')
plt.xlabel('V_FDTD')
plt.ylabel('V_NN')
plt.xlim(0.75, 1.5)
plt.ylim(0.75, 1.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('D:/ERG4901/ViT_Project/Vcorr_coeff_train.eps')
plt.show()

"""V coefficient test"""
testQ_loss, testV_loss, Q_out, Q_tarout, V_out, V_tarout = test_epoch(modelV, test_loader)
print('\nBest V: Vloss: {:.6f}\n'.format(testV_loss))


Q_out = torch.cat(Q_out, 0)
Q_tarout = torch.cat(Q_tarout, 0)
V_out = torch.cat(V_out, 0)
V_tarout = torch.cat(V_tarout, 0)
Q_outArr = Q_out.cpu().numpy()  # for Q
Q_taroutArr = Q_tarout.cpu().numpy()
V_outArr = V_out.cpu().numpy()  # for V
V_taroutArr = V_tarout.cpu().numpy()

coeffV_test_best = pearson_r(V_outArr, V_taroutArr)
print('Best test V corr coeff: {:.3f}'.format(coeffV_test_best))
coeffV_test_best = round(coeffV_test_best, 3)

fig = plt.figure()
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.scatter(V_taroutArr, V_outArr, s=5, color='purple')
plt.legend(['Test data (2500)\ncorrelation coeff ' + "{:.3f}".format(coeffV_test_best)], loc='upper right')
x = np.linspace(0, 1.5, 5)
y = x
plt.plot(x, y, c='k')
plt.xlabel('V_FDTD')
plt.ylabel('V_NN')
plt.axis('square')
plt.xlim(0.75, 1.5)
plt.ylim(0.75, 1.5)
plt.savefig('D:/ERG4901/ViT_Project/Vcorr_coeff_test.eps')
plt.show()

# %%

