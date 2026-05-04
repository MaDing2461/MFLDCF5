import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert(np.max(pred) < self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    @staticmethod
    def _safe_divide(numerator, denominator):
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self._safe_divide(self.M[i, i], np.sum(self.M[i, :]))

        return recall/self.nclass

    def precision(self):
        precision = 0.0
        for i in range(self.nclass):
            precision += self._safe_divide(self.M[i, i], np.sum(self.M[:, i]))

        return precision/self.nclass

    # def recall(self):
    #     recall = 0.0
    #     for i in range(self.nclass):
    #         if i > 0:
    #             recall += self.M[i, i] / np.sum(self.M[:, i])

    #     return recall/1

    def accuracy(self):
        return self._safe_divide(np.sum(np.diag(self.M)), np.sum(self.M))

    # def accuracy(self):
    #     accuracy = 0.0
    #     for i in range(self.nclass):
    #         if i > 0:
    #             accuracy += self.M[i, i] / np.sum(self.M[i, :])

    #     return accuracy/1

    def jaccard(self):
        jaccard_perclass = []
        for i in range(self.nclass):
            denominator = np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]
            jaccard_perclass.append(self._safe_divide(self.M[i, i], denominator))

        return np.mean(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        gt = gt.astype(np.int64)
        pred = pred.astype(np.int64)
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if 0 <= gt[i] < self.nclass and 0 <= pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m
    
    def f1_score(self):  
        f1_scores = []  
        for i in range(self.nclass):  
            # 避免除以零的情况  
            if (self.M[i, i] == 0) or (np.sum(self.M[:, i]) == 0) or (np.sum(self.M[i, :]) == 0):  
                f1_scores.append(0.0)  
                continue  
              
            precision = self.M[i, i] / np.sum(self.M[:, i])  
            recall = self.M[i, i] / np.sum(self.M[i, :])  
            f1 = 2 * (precision * recall) / (precision + recall)  
            f1_scores.append(f1)  
  
        # 返回平均F1分数和每个类别的F1分数  
        return np.mean(f1_scores), f1_scores  

    # def f1_score(self):  
    #     f1_scores = []  
    #     for i in range(self.nclass):  
    #         # 避免除以零的情况  
    #         if (self.M[i, i] == 0) or (np.sum(self.M[:, i]) == 0) or (np.sum(self.M[i, :]) == 0):  
    #             f1_scores.append(0.0)  
    #             continue  
              
    #         precision = self.M[i, i] / np.sum(self.M[i, :])  
    #         recall = self.M[i, i] / np.sum(self.M[:, i])  
    #         f1 = 2 * (precision * recall) / (precision + recall)  
    #         f1_scores.append(f1)  
  
    #     # 返回平均F1分数和每个类别的F1分数  
    #     return f1_scores[1], f1_scores  

def get_iou(data_list, class_num, save_path=None):
    if(len(data_list)==0):
        return 
    from multiprocessing import Pool
	
    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    recall = ConfM.recall()
    precision = ConfM.precision()
    acc = ConfM.accuracy()
    mean_f1, f1 = ConfM.f1_score()

    default_classes = ('fake', 'real')
    classes = np.array([
        default_classes[i] if i < len(default_classes) else 'class_{}'.format(i)
        for i in range(class_num)
    ])

    for i in range(class_num):
        print('class {:2d} {:12} IU {:.4f} F1 {:.4f}'.format(i, classes[i], j_list[i],f1[i]))
    print('meanIOU: {:.4f} Recall: {:.4f} Precision: {:.4f} Accuracy: {:.4f} F1: {:.4f}'.format(aveJ, recall, precision, acc, mean_f1))

    return aveJ


def get_Acc(data_real,data_pre):
    test_image_labels = data_real
    test_p = data_pre
    f1 = f1_score(test_image_labels, test_p)
    acc = accuracy_score(test_image_labels, test_p)
    precision = precision_score(test_image_labels, test_p, zero_division=0)
    recall = recall_score(test_image_labels, test_p, zero_division=0)
    print("2026.4.2")
    print("Image F1 score: {:.4f} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f}". format(f1,acc,precision, recall))
    return acc

def get_F1(data_real,data_pre):
    test_image_labels = data_real
    test_p = data_pre
    f1 = f1_score(test_image_labels, test_p)
    return f1
