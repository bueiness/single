import pandas as pd
from pandas import DataFrame
import jieba
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics


def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))


def make_label(df):
    df["sentiment"] = df["star"].apply(lambda x: 1 if x > 3 else 0)


def get_custom_stopword(stop_word_file):
    with open(stop_word_file, encoding='utf-8') as f:
        stop_word = f.read()
    stop_word_list = stop_word.split("/n")
    custom_stopword = [i for i in stop_word_list]
    return custom_stopword


if __name__ == '__main__':
    # 读取文件并配置标签
    data = pd.read_csv("data.csv", encoding='GB18030')
    make_label(data)
    # 获取评论列以及标签列
    X = data[["comment"]]
    y = data.sentiment
    # 对评论列内容分词且创造训练集以及测试集
    X["cuted_comment"] = X.comment.apply(chinese_word_cut)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # 去除停用词、过于普通和过于特殊的词
    stopwords = get_custom_stopword("哈工大停用词表.txt")
    vect = CountVectorizer(max_df=0.8,
                           min_df=3,
                           token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                           stop_words=frozenset(stopwords))
    term_matrix = DataFrame(vect.fit_transform(X_train.cuted_comment).toarray(), columns=vect.get_feature_names())
    # 导入朴素贝叶斯函数，创建分类模型
    nb = MultinomialNB()
    pipe = make_pipeline(vect, nb)
    # 将未特征向量化的数据导入，提高模型准确率
    cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
    pipe.fit(X_train.cuted_comment, y_train)
    # 获得预测结果
    y_pred = pipe.predict(X_test.cuted_comment)
    # 比较预测结果与测试结果获得准确率
    print(metrics.accuracy_score(y_test, y_pred))
