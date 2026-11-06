import jieba
text = "小明毕业于北京大学计算机系"

def demo1():
    print(jieba.lcut(text))
    # ['小明', '毕业', '于', '北京大学', '计算机', '系']

    print(jieba.lcut(text, cut_all=True))
    # ['小', '小明', '明', '毕业', '于', '北京', '北京大学', '大学', '计算机', '算机', '系']

    print(jieba.lcut_for_search(text))
    # ['小明', '毕业', '于', '北京', '大学', '北京大学', '计算机', '系']

def demo2():
    text = "使用深度学习进行图像分割, 我喜欢深度学习和机器学习"

    jieba.load_userdict("my_words_dict.txt")

    

    print(jieba.lcut(text))
    # ['使用', '深度', '学习', '进行', '图像', '分割']
    # 问题：“深度学习”“图像分割”被拆开了——通用词表不认识领域词


def demo3():
    print("\n-- 示例 3: 停用词过滤")

    text = "我是一个学生，我正在使用jieba进行中文分词，它是一个非常强大的工具！"

    # 1. 先分词
    words = jieba.lcut(text)
    print("分词结果:", words)

    # 2. 定义停用词表（实际项目中通常从 stopwords.txt 文件读取，每行一个词）
    stopwords = {"我", "是", "一个", "正在", "进行", "它", "的", "非常"}

    # 3. 过滤：去掉停用词 + 标点符号 + 空白
    filtered = [
        w for w in words
        if w not in stopwords and w.strip() and w not in "，。！？、；：""''"
    ]
    print("过滤后:", filtered)
    # ['学生', '使用', 'jieba', '中文', '分词', '强大', '工具']


if __name__ == '__main__':
    demo3()