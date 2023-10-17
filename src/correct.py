from textblob import TextBlob

# 所有需要修改的文件
file_paths = [
    "./data/ISEAR ID_n",
    "./data/ISEAR ID_test_n",
    "./data/ISEAR ID_train_n",
    "./data/ISEAR ID_validation_n"
]

# 去重字典
dic = {}

def correct(sentence):
    '''
    修改给定句子, 如果给定句子已经出现过,
    则返回跳过信号
    '''
    if sentence in dic:
        return sentence, True
    else:
        dic[sentence] = 1
        textBlb = TextBlob(sentence)            
        return textBlb.correct().__str__(), False

def update(file):
    '''
    清理数据，包括改错词和去重
    '''
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        j = 0
        line_list = list(f)
        for line in line_list:
            j += 1
            print(j, len(line_list))
            new_line = ""
            word_list = line.split('\t')
            skip = False
            for i in range(len(word_list)):
                if i == len(word_list) - 1:
                    new_word, skip = correct(word_list[i])
                    new_line = new_line + new_word 
                else:
                    new_line = new_line + word_list[i] + '\t'
            if not skip:
                file_data += new_line

    with open(file + "_f", "w" , encoding="utf-8") as f:
        f.write(file_data)

for file in file_paths:
    dic.clear()
    update(file)