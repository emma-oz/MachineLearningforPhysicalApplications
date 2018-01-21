"""
    ==================================================
    Plot different SVM classifiers for Noise09
    by Haiqiang Niu 03/14/2017
    Modified to do SBCEx16 by Emma Ozanich 01/16/2018
    ==================================================
    
    """
datapath = '../data/'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# Training Data
X_train = np.loadtxt( datapath + 'DataSet01/train_input/SBCEx16_train_input.txt')
Y_train = np.loadtxt( datapath + 'DataSet01/train_label/SBCEx16_train_label.txt')
Range_train = np.loadtxt( datapath + 'DataSet01/Mapping_range_labels.txt')

Y_train = [np.where(c==1)[0][0] for c in Y_train]

# Test Data
X_test = np.loadtxt( datapath + 'DataSet01/test_input/SBCEx16_test_input.txt')
Y_test = np.loadtxt( datapath + 'DataSet01/test_Ranges.txt')

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear').fit(X_train, Y_train)
print('SVM classification with linear kernel is done!')
rbf_svc = svm.SVC(kernel='rbf', gamma=1/149.0, C=C).fit(X_train, Y_train)
print('SVM classification with radial basis function kernel is done!')
poly_svc = svm.SVC(kernel='poly', gamma=1/149.0,degree=1, C=C).fit(X_train, Y_train)
print('SVM classification with polynomial kernel is done!')
lin_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
print('Linear SVM classification is done!')

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC',
          'SVC with RBF kernel (gamma=1/149)',
          'SVC with polynomial (degree 1) kernel']

fig=plt.figure(figsize=(5.0,4.0))
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(X_test)
    Z_out = Range_train[Z.astype(np.int32)]
    
    acc = np.logical_and([Z_out >= Y_test*0.9],[Z_out <= Y_test*1.1])
    acc = np.divide(sum(sum(acc)),float(len(Y_test)))*100
    print(titles[i] + ' predicts the range with ' + ' %0.1f %% accuracy, within 10 %% error.' % acc)

    plt.plot(Z_out,"o",markersize=2,markeredgewidth=0.5,markeredgecolor='b',markerfacecolor='none')
    plt.plot(Y_test,'r',linewidth=1.0)

    plt.xlabel('Time [index]',fontsize=8)
    plt.ylabel('Range (m)',fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.title(titles[i],fontsize=8)


plt.show()
fig.savefig('Fig_DataSet01_SVM.jpg', dpi=300)
