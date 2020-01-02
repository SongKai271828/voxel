from mylib.models.misc import set_gpu_usage

set_gpu_usage()

from mylib.models import densenet
import utils
import numpy as np
from Global import *

# 导入测试集时的参数
BATCH_SIZE_TEST = 3
DEAL_SIZE = 32

# 定义测试集类
test_set = utils.Dataset(
    data_path=test_path,
    label_path=test_label_path,
    batch=BATCH_SIZE_TEST,
    type='test',
    pre=True,
    deal_size=DEAL_SIZE,
    enhance=False,
)

model = densenet.get_compiled()

test_data, test_label = test_set.load_all()

# 使用模型编号的列表
item = [0]

# 多个模型推断时每个模型的权重
lam = [1]


def load_and_eval():
    results = np.zeros([117, ], dtype=np.float)
    names = ''
    for name in range(len(item)):
        model.load_weights(good_path+str(item[name])+'.h5')
        result = model.predict(x=test_data, batch_size=BATCH_SIZE_TEST, verbose=1)
        result1 = result[:, 1]
        np.savetxt(answers_path + str(item[name]) + '_a' + '.csv', result1, delimiter=',')
	# 综合推断结果        
	results = results + result1 * lam[name]
        names += str(item[name]) + '_'
    # 保存推断结果
    np.savetxt('result_' + names + '.csv', results, delimiter=',')


if __name__ == '__main__':
    load_and_eval()
