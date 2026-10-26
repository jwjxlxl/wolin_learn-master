""" 🎯 趣味引入：给一段文字生成词云
分词 → 统计词频 → 画成词云，直观看到"哪些词最重要"
需要安装: pip install jieba wordcloud matplotlib
"""
import jieba
import wordcloud
import matplotlib.pyplot as plt
from collections import Counter

# 1. 准备⼀段⽂本
text = """
    ⼈⼯智能是计算机科学的⼀个分⽀，它试图了解智能的实质，
    并⽣产出⼀种新的能以⼈类智能相似的⽅式做出反应的智能机器。
    ⼈⼯智能的研究包括机器⼈、语⾔识别、图像识别、⾃然语⾔处理和专家系统等。
    深度学习是机器学习的⼀个分⽀，神经⽹络是深度学习的核⼼。
    卷积神经⽹络在图像识别中取得了突破性成果，
    Transformer 架构在⾃然语⾔处理中引发了⾰命性的变化。
    ⼤语⾔模型如 GPT、Claude 等正在改变⼈机交互的⽅式。
    机器学习算法包括监督学习、⽆监督学习和强化学习。
    计算机视觉让机器能够看懂世界，⾃然语⾔处理让机器能够读懂⽂字。
    ⼈⼯智能的应⽤正在改变医疗、⾦融、教育、交通等各⾏各业。
"""
# 2. 分词 + 过滤
words = jieba.lcut(text)
# 去掉单字和停⽤词
stop_words = {"的", "是", "⼀个", "并", "等", "在", "了",
              "中", "和", "让", "能够", "等。"}
meaningful = [w for w in words if len(w) > 1 and w not in stop_words]
# 3. 统计词频
freq = Counter(meaningful)
print("📊 词频 Top 10:")
for word, count in freq.most_common(10):
    bar = "█" * count
    print(f" {word:8s} {count:2d} {bar}")
# 4. ⽣成词云
wc = wordcloud.WordCloud(font_path="msyh.ttc",  # Windows ⾃带字体
                         width=800, height=400,
                         background_color="white",
                         max_words=50,
                         max_font_size=80,
                         colormap="viridis",
                         )
wc.generate_from_frequencies(freq)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("AI 领域关键词云", fontsize=16, fontproperties=plt.rcParams["font.sans-serif"][0])
plt.tight_layout()
plt.savefig("ai_wordcloud.png", dpi=150)
print("\n✅ 词云已保存为 ai_wordcloud.png")
plt.show()
