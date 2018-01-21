"""
==================================================
Plot random forest classifier for Noise09
by Haiqiang Niu 03/21/2017
Modified to use SBCEx16 by Emma Ozanich, 01/16/2018
==================================================

"""
print(__doc__)

datapath = '../data/'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Train Data
X_train = np.loadtxt( datapath + 'DataSet01/train_input/SBCEx16_train_input.txt')
Y_train = np.loadtxt( datapath + 'DataSet01/train_label/SBCEx16_train_label.txt')
Range_train = np.loadtxt( datapath + 'DataSet01/Mapping_range_labels.txt')

Y_train = [np.where(c==1)[0][0] for c in Y_train]
np.savetxt('testing_inputs',Y_train)

# Test Data
X_test = np.loadtxt( datapath + 'DataSet01/test_input/SBCEx16_test_input.txt')
Y_test = np.loadtxt( datapath + 'DataSet01/test_Ranges.txt')

k = 3
clf1 = RandomForestClassifier(n_estimators=100,n_jobs=k,min_samples_leaf=50,verbose=False).fit(X_train, Y_train)
print('Training process for random forest with 100 estimators done!')

clf2 = RandomForestClassifier(n_estimators=200,n_jobs=k,min_samples_leaf=50,verbose=False).fit(X_train, Y_train)
print('Training process for random forest with 200 estimators done!')

clf3 = RandomForestClassifier(n_estimators=500,n_jobs=k,min_samples_leaf=50,verbose=False).fit(X_train, Y_train)
print('Training process for random forest with 500 estimators done!')

clf4 = RandomForestClassifier(n_estimators=1000,n_jobs=k,min_samples_leaf=50,verbose=False).fit(X_train, Y_train)
print('Training process for random forest with 1000 estimators done!')

# title for the plots
titles = ['Random forest with n_estimators=100',
          'Random forest with n_estimators=200',
          'Random forest with n_estimators=500',
          'Random forest with n_estimators=1000']

fig=plt.figure(figsize=(5.0,4.0))
for i, clf in enumerate((clf1, clf2, clf3, clf4)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(X_test)
    Z_out = Range_train[Z.astype(np.int32)]
    
    acc = np.logical_and([Z_out >= Y_test*0.9],[Z_out <= Y_test*1.1])
    acc = np.divide(sum(sum(acc)),float(len(Y_test)))*100
    print(titles[i] + ' predicts the range with ' + ' %0.1f %% accuracy, within 10 %% error.' % acc)
    
    plt.plot(Z_out,'o',color='blue',markersize=2,mew=0.5,mec='blue',markerfacecolor='none')
    plt.plot(Y_test,'r',linewidth=1.0)

    plt.xlabel('Time [index]',fontsize=8)
    plt.ylabel('Range (km)',fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)

    plt.title(titles[i],fontsize=8)


plt.show()
fig.savefig('Fig_DataSet01_RF.jpg', dpi=300)

