import re
import jieba
def tokens(text):
    return [w for w in jieba.lcut(text) if re.search(r'[\w]',w)]
