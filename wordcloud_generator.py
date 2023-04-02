from wordcloud import WordCloud, STOPWORDS #wordcloud-1.8.2.2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# define the punctuation and stopwords to be removed
removable_punctuation = '",:;?!()-.'
stopwords = set(STOPWORDS)

def make_wordcloud(max_words : int = 250, min_font_size : int = 5, bg_color : str = 'black', colormap: str = 'viridis', circle : bool = True):
    """
    This function creates and image of a word cloud for all news articles in the data.

    - max_words: the maximum number of words to be displayed in the word cloud
    - min_font_size: the minimum font size for the words in the word cloud
    - bg_color: background color of the word cloud
    - colormap: colormap for the words in the word cloud
    - circle: whether to make a circular word cloud
    """
    
    counter = Counter()
        
    for file in range(845):
        path = 'data/articles/' + str(file) + '.txt'
        
        # for each file, compute the frequencies of the words
        # remove punctuation and stopwords
        # store everything in 1 counter
        with open(path) as f:
            words = f.read().split()
            words_lower = [word.lower() for word in words]
            no_punc = [''.join(char for char in word if char not in removable_punctuation) for word in words_lower]
            no_punc = list(filter(None, no_punc))
            interesting_words = [word for word in no_punc if word not in stopwords]
            counter = counter + Counter(interesting_words)
    
    # create a circular word cloud
    if circle:
        x, y = np.ogrid[:800, :800]
        mask = (x - 400) ** 2 + (y - 400) ** 2 > 400 ** 2
        mask = 255 * mask.astype(int)

        wordcloud = WordCloud(width = 800, height = 800,
                        background_color = bg_color,
                        colormap = colormap,
                        stopwords = stopwords,
                        mask = mask,
                        max_words = max_words,
                        min_font_size = min_font_size, 
                        random_state=0).fit_words(counter)
    
    # create a square word cloud
    else:
        wordcloud = WordCloud(width = 800, height = 800,
                        background_color = bg_color,
                        colormap = colormap,
                        stopwords = stopwords,
                        max_words = max_words,
                        min_font_size = min_font_size, 
                        random_state=0).fit_words(counter)
    
    # plot the WordCloud image                      
    plt.figure(figsize = (8, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('assets/wordcloud.png')
make_wordcloud(250, 5, '#26232C', 'viridis', True)