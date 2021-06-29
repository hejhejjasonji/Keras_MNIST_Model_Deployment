# from keras.models import load_model
from DeepMnist import DeepMnist
from MnistClassifier import MnistClassifier
# # import json
# # with open('./contract.json','r',encoding='utf8')as fp:
# #     json_data = json.load(fp)
# #     print('这是文件中的json数据：',json_data)
# #     print('这是读取到文件数据的数据类型：', type(json_data))
import pickle as pkl
with open("D:\wisers\dataset_of_mnist\one_record.pkl","rb") as fh:
    b=pkl.load(fh)
# mc=MnistClassifier()
mc=DeepMnist()
features = ["X"+str(i+1) for i in range (0,784)]
print(mc.predict(b,features))
# load_model('MnistClassifier.h5').predict(b.tolist())