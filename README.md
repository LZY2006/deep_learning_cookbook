# Deep Learning Cookbook Notebooks

该存储库包含35个python笔记本，展示了大部分关键
Keras中的机器学习技术。 笔记本随书
[Deep Learning Cookbook](https://www.amazon.com/Deep-Learning-Cookbook-Practical-Recipes) 但只看书就很好了， 不需要GPU就可以运行这些笔记本，
但是仅使用CPU会花费相当长的时间。

## 准备工作

首先，设置虚拟环境，安装环境要求，然后启动笔记本服务器：

```Bash
git clone https://github.com/DOsinga/deep_learning_cookbook.git
cd deep_learning_cookbook
python3 -m venv venv3
source venv3/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## 包含的笔记本

<<<<<<< HEAD
#### [03.1 Using pre trained word embeddings](03.1 Using pre trained word embeddings.ipynb)

在本笔记本中，我们使用预训练的单词嵌入模型（Word2Vec）来使用单词嵌入来探索单词之间的相似性以及单词之间的关系。 例如，找到一个国家的首都或公司的主要产品。 我们会使用t-SNE在2D图形上绘制高维空间，说明内部原理。

#### [03.2 Domain specific ranking using word2vec cosine distance](03.2 Domain specific ranking using word2vec cosine distance.ipynb)

在前面的的基cookbook的基础上，我们会用单词之间的距离进行特定领域的排名。 具体来说，我们将研究国家。 首先，我们基于小样本创建小分类器，然后在单词集中找到所有国家。 然后，我们用类似的方法，展示对各国单词的相关性。 例如，由于板球比德国更接近印度，板球可能是更相关。 我们可以在世界地图画出各个国家的相关性。

#### [04.1 Collect movie data from Wikipedia](04.1 Collect movie data from Wikipedia.ipynb)

本笔记本展示了如何把维基百科的数据下载导出文件，并使用类别和模板信息解析、提取、结构化数据。 我们用数据创建一组包含评级数据的电影信息。

#### [04.2 Build a recommender system based on outgoing Wikipedia links](04.2 Build a recommender system based on outgoing Wikipedia links.ipynb)

基于上一本笔记本中提取的结构化数据，我们会训练一个神经网络，该网络根据相应的维基百科页面的链接预测电影的评级。我们会为电影创建嵌入。 反过来，我们可以根据其他电影来推荐电影的嵌入空间中相似的电影。

#### [05.1 Generating Text in the Style of an Example Text](05.1 Generating Text in the Style of an Example Text.ipynb)

我们会训练一个循环神经网络（LSTM）来撰写类似莎士比亚文本。 然后，我们会用Python标准库上的代码训练类似的LSTM来生成Python代码。 可视化网络，探索神经网络在生产或读取Python代码时在注意些什么。

#### [06.1 Question matching](06.1 Question matching.ipynb)

在本笔记本中，我们训练了一个网络来学习如何匹配来自stackoverflow的问题和答案。 这种索引使我们能够找到给定的问题，数据库中最可能的答案是什么。 我们尝试了多种方法来改善最初并非十分出色的结果。

#### [07.1 Text Classification](07.1 Text Classification.ipynb)

该笔记本显示了八种不同的机器学习方法，可将文本分为各种情感。 前三个是古典学习者，其次是许多深度学习模型，基于字符或单词的学习以及lstm vs cnn。 最好的方法是将所有方法组合到一个模型中。

#### [07.2 Emoji Suggestions](07.2 Emoji Suggestions.ipynb)

我们首先收集大量推文，然后保留仅包含一个表情符号的推文（您可以跳过此步骤，其中包括一个训练集）。 然后，我们训练许多深度模型，以使用推特减去表情符号来预测缺失的表情符号。 我们最终以一个可以为给定文本找到最佳表情符号的模型作为最终结果。

#### [07.3 Tweet Embeddings](07.3 Tweet Embeddings.ipynb)

一些实验性代码（书中未包含）可在语义上对推文进行索引，以使相似的推文彼此相邻显示； 有效地执行Word2Vec针对单词所做的操作，但现在针对推文执行。

#### [08.1 Sequence to sequence mapping](08.1 Sequence to sequence mapping.ipynb)
#### [08.2 Import Gutenberg](08.2 Import Gutenberg.ipynb)

演示如何从Gutenberg项目下载书籍的小笔记本。 在准备下一个笔记本中的子词标记化时，对一组书籍进行标记化。

#### [09.1 Reusing a pretrained image recognition network](09.1 Reusing a pretrained image recognition network.ipynb)

快速笔记本演示了如何加载预训练的网络并将其应用到其他图像上？ 一只猫。 显示如何规范化图像并解码预测。

#### [09.2 Images as embeddings](09.2 Images as embeddings.ipynb)

在此笔记本中，我们使用Flickr API来获取搜索项cat的搜索结果供稿。 通过经过预训练的网络运行每个结果，我们得到将图像投影在“空间”中的向量。 该空间的中心以某种方式代表了最多的猫图像。 通过对到该中心的距离重新排序，我们可以剔除不太像猫的图像。 有效地，我们可以在不知道内容的情况下改善Flickr的搜索结果！

#### [09.3 Retraining](09.3 Retraining.ipynb)
#### [10.1 Building an inverse image search service](10.1 Building an inverse image search service.ipynb)
#### [11.1 Detecting Multiple Images](11.1 Detecting Multiple Images.ipynb)

利用imag分类网络为每个较大的方形子图像提取特征的事实，以检测同一图像中的多只猫狗，或者至少知道在图像中的什么位置可以找到猫或狗。 她的方法比最新技术要简单得多，但也容易遵循，因此是入门的好方法。

#### [12.1 Activation Optimization](12.1 Activation Optimization.ipynb)
#### [12.2 Neural Style](12.2 Neural Style.ipynb)
#### [13.1 Quick Draw Cat Autoencoder](13.1 Quick Draw Cat Autoencoder.ipynb)
#### [13.2 Variational Autoencoder](13.2 Variational Autoencoder.ipynb)
#### [13.5 Quick Draw Autoencoder](13.5 Quick Draw Autoencoder.ipynb)
#### [14.1 Importing icons](14.1 Importing icons.ipynb)
#### [14.2 Icon Autoencoding](14.2 Icon Autoencoding.ipynb)
#### [14.2 Variational Autoencoder Icons](14.2 Variational Autoencoder Icons.ipynb)
#### [14.3 Icon GAN](14.3 Icon GAN.ipynb)
#### [14.4 Icon RNN](14.4 Icon RNN.ipynb)
#### [15.1 Song Classification](15.1 Song Classification.ipynb)
#### [15.2 Index Local MP3s](15.2 Index Local MP3s.ipynb)
#### [15.3 Spotify Playlists](15.3 Spotify Playlists.ipynb)
#### [15.4 Train a music recommender](15.4 Train a music recommender.ipynb)
#### [16.1 Productionize Embeddings](16.1 Productionize Embeddings.ipynb)
#### [16.2 Prepare Keras model for Tensorflow Serving](16.2 Prepare Keras model for Tensorflow Serving.ipynb)
#### [16.3 Prepare model for iOS](16.3 Prepare model for iOS.ipynb)
#### [16.4 Simple Text Generation](16.4 Simple Text Generation.ipynb)
#### [Simple Seq2Seq](Simple Seq2Seq.ipynb)
=======
#### [03.1 Using pre trained word embeddings](https://github.com/DOsinga/deep_learning_cookbook/blob/master/03.1%20Using%20pre%20trained%20word%20embeddings.ipynb)

在本笔记本中，我们使用预训练的单词嵌入模型（Word2Vec）来使用单词嵌入来探索单词之间的相似性以及单词之间的关系。 例如，找到一个国家的首都或公司的主要产品。 我们会使用t-SNE在2D图形上绘制高维空间，说明内部原理。

#### [03.2 Domain specific ranking using word2vec cosine distance](https://github.com/DOsinga/deep_learning_cookbook/blob/master/03.2%20Domain%20specific%20ranking%20using%20word2vec%20cosine%20distance.ipynb)

在前面的的基cookbook的基础上，我们会用单词之间的距离进行特定领域的排名。 具体来说，我们将研究国家。 首先，我们基于小样本创建小分类器，然后在单词集中找到所有国家。 然后，我们用类似的方法，展示对各国单词的相关性。 例如，由于板球比德国更接近印度，板球可能是更相关。 我们可以在世界地图画出各个国家的相关性。

#### [04.1 Collect movie data from Wikipedia](https://github.com/DOsinga/deep_learning_cookbook/blob/master/04.1%20Collect%20movie%20data%20from%20Wikipedia.ipynb)

本笔记本展示了如何把维基百科的数据下载导出文件，并使用类别和模板信息解析、提取、结构化数据。 我们用数据创建一组包含评级数据的电影信息。

#### [04.2 Build a recommender system based on outgoing Wikipedia links](https://github.com/DOsinga/deep_learning_cookbook/blob/master/04.2%20Build%20a%20recommender%20system%20based%20on%20outgoing%20Wikipedia%20links.ipynb)

基于上一本笔记本中提取的结构化数据，我们会训练一个神经网络，该网络根据相应的维基百科页面的链接预测电影的评级。我们会为电影创建嵌入。 反过来，我们可以根据其他电影来推荐电影的嵌入空间中相似的电影。

#### [05.1 Generating Text in the Style of an Example Text](https://github.com/DOsinga/deep_learning_cookbook/blob/master/05.1%20Generating%20Text%20in%20the%20Style%20of%20an%20Example%20Text.ipynb)

我们会训练一个循环神经网络（LSTM）来撰写类似莎士比亚文本。 然后，我们会用Python标准库上的代码训练类似的LSTM来生成Python代码。 可视化网络，探索神经网络在生产或读取Python代码时在注意些什么。

#### [06.1 Question matching](https://github.com/DOsinga/deep_learning_cookbook/blob/master/06.1%20Question%20matching.ipynb)

在本笔记本中，我们训练了一个网络来学习如何匹配来自stackoverflow的问题和答案。 这种索引使我们能够找到给定的问题，数据库中最可能的答案是什么。 我们尝试了多种方法来改善最初并非十分出色的结果。

#### [07.1 Text Classification](https://github.com/DOsinga/deep_learning_cookbook/blob/master/07.1%20Text%20Classification.ipynb)

该笔记本显示了八种不同的机器学习方法，可将文本分为各种情感。 前三个是古典学习者，其次是许多深度学习模型，基于字符或单词的学习以及lstm vs cnn。 最好的方法是将所有方法组合到一个模型中。

#### [07.2 Emoji Suggestions](https://github.com/DOsinga/deep_learning_cookbook/blob/master/07.2%20Emoji%20Suggestions.ipynb)

我们首先收集大量推文，然后保留仅包含一个表情符号的推文（您可以跳过此步骤，其中包括一个训练集）。 然后，我们训练许多深度模型，以使用推特减去表情符号来预测缺失的表情符号。 我们最终以一个可以为给定文本找到最佳表情符号的模型作为最终结果。

#### [07.3 Tweet Embeddings](https://github.com/DOsinga/deep_learning_cookbook/blob/master/07.3%20Tweet%20Embeddings.ipynb)

一些实验性代码（书中未包含）可在语义上对推文进行索引，以使相似的推文彼此相邻显示； 有效地执行Word2Vec针对单词所做的操作，但现在针对推文执行。

#### [08.1 Sequence to sequence mapping](https://github.com/DOsinga/deep_learning_cookbook/blob/master/08.1%20Sequence%20to%20sequence%20mapping.ipynb)
#### [08.2 Import Gutenberg](https://github.com/DOsinga/deep_learning_cookbook/blob/master/08.2%20Import%20Gutenberg.ipynb)

演示如何从Gutenberg项目下载书籍的小笔记本。 在准备下一个笔记本中的子词标记化时，对一组书籍进行标记化。

#### [09.1 Reusing a pretrained image recognition network](https://github.com/DOsinga/deep_learning_cookbook/blob/master/09.1%20Reusing%20a%20pretrained%20image%20recognition%20network.ipynb)

快速笔记本演示了如何加载预训练的网络并将其应用到其他图像上？ 一只猫。 显示如何规范化图像并解码预测。

#### [09.2 Images as embeddings](https://github.com/DOsinga/deep_learning_cookbook/blob/master/09.2%20Images%20as%20embeddings.ipynb)

在此笔记本中，我们使用Flickr API来获取搜索项cat的搜索结果供稿。 通过经过预训练的网络运行每个结果，我们得到将图像投影在“空间”中的向量。 该空间的中心以某种方式代表了最多的猫图像。 通过对到该中心的距离重新排序，我们可以剔除不太像猫的图像。 有效地，我们可以在不知道内容的情况下改善Flickr的搜索结果！

#### [09.3 Retraining](https://github.com/DOsinga/deep_learning_cookbook/blob/master/09.3%20Retraining.ipynb)
#### [10.1 Building an inverse image search service](https://github.com/DOsinga/deep_learning_cookbook/blob/master/10.1%20Building%20an%20inverse%20image%20search%20service.ipynb)
#### [11.1 Detecting Multiple Images](https://github.com/DOsinga/deep_learning_cookbook/blob/master/11.1%20Detecting%20Multiple%20Images.ipynb)

利用imag分类网络为每个较大的方形子图像提取特征的事实，以检测同一图像中的多只猫狗，或者至少知道在图像中的什么位置可以找到猫或狗。 她的方法比最新技术要简单得多，但也容易遵循，因此是入门的好方法。

#### [12.1 Activation Optimization](https://github.com/DOsinga/deep_learning_cookbook/blob/master/12.1%20Activation%20Optimization.ipynb)
#### [12.2 Neural Style](https://github.com/DOsinga/deep_learning_cookbook/blob/master/12.2%20Neural%20Style.ipynb)
#### [13.1 Quick Draw Cat Autoencoder](https://github.com/DOsinga/deep_learning_cookbook/blob/master/13.1%20Quick%20Draw%20Cat%20Autoencoder.ipynb)
#### [13.2 Variational Autoencoder](https://github.com/DOsinga/deep_learning_cookbook/blob/master/13.2%20Variational%20Autoencoder.ipynb)
#### [13.5 Quick Draw Autoencoder](https://github.com/DOsinga/deep_learning_cookbook/blob/master/13.5%20Quick%20Draw%20Autoencoder.ipynb)
#### [14.1 Importing icons](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.1%20Importing%20icons.ipynb)
#### [14.2 Icon Autoencoding](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.2%20Icon%20Autoencoding.ipynb)
#### [14.2 Variational Autoencoder Icons](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.2%20Variational%20Autoencoder%20Icons.ipynb)
#### [14.3 Icon GAN](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.3%20Icon%20GAN.ipynb)
#### [14.4 Icon RNN](https://github.com/DOsinga/deep_learning_cookbook/blob/master/14.4%20Icon%20RNN.ipynb)
#### [15.1 Song Classification](https://github.com/DOsinga/deep_learning_cookbook/blob/master/15.1%20Song%20Classification.ipynb)
#### [15.2 Index Local MP3s](https://github.com/DOsinga/deep_learning_cookbook/blob/master/15.2%20Index%20Local%20MP3s.ipynb)
#### [15.3 Spotify Playlists](https://github.com/DOsinga/deep_learning_cookbook/blob/master/15.3%20Spotify%20Playlists.ipynb)
#### [15.4 Train a music recommender](https://github.com/DOsinga/deep_learning_cookbook/blob/master/15.4%20Train%20a%20music%20recommender.ipynb)
#### [16.1 Productionize Embeddings](https://github.com/DOsinga/deep_learning_cookbook/blob/master/16.1%20Productionize%20Embeddings.ipynb)
#### [16.2 Prepare Keras model for Tensorflow Serving](https://github.com/DOsinga/deep_learning_cookbook/blob/master/16.2%20Prepare%20Keras%20model%20for%20Tensorflow%20Serving.ipynb)
#### [16.3 Prepare model for iOS](https://github.com/DOsinga/deep_learning_cookbook/blob/master/16.3%20Prepare%20model%20for%20iOS.ipynb)
#### [16.4 Simple Text Generation](https://github.com/DOsinga/deep_learning_cookbook/blob/master/16.4%20Simple%20Text%20Generation.ipynb)
#### [Simple Seq2Seq](https://github.com/DOsinga/deep_learning_cookbook/blob/master/Simple%20Seq2Seq.ipynb)
>>>>>>> parent of 1fb2334... 修复链接格式

简单的序列到序列映射演示。 笔记本显示了如何教网络如何形成复数。
