Written for Python 3.6 (Jan 2018)

'localization_RF.py' and 'localization_SVM.py' are stand-alone.

For FNN:

1. Tensorflow MUST be installed properly with current version of Python. For instructions, see (*): https://www.tensorflow.org/install/install_mac.

2. Run 'localization_with_MultiFreq.py'. It will call 'load_data_nhq_si.py' automatically. Be sure data path is set correctly.


(*) For further help with Conda, see: https://github.com/conda-forge/tensorflow-feedstock. Note, the Tensorflow binaries are buggy for certain builds. If you encounter a bug, visit GitHub to find updated binaries (e.g. https://github.com/lakshayg/tensorflow-build).