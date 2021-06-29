from keras.models import load_model

class MnistClassifier(object):

    def __init__(self):
        self.model = load_model('MnistClassifier.h5')
        self.model._make_predict_function()

    def predict(self,X,features_names):
        return self.model.predict(X)
# import pickle as pkl
# with open("D:\wisers\dataset_of_mnist\one_record.pkl","rb") as fh:
#     b=pkl.load(fh)
# mc=MnistClassifier
# features = ["X"+str(i+1) for i in range (0,784)]
# print(mc.predict(self,X=b.tolist(),features_names=features))
# print(b)