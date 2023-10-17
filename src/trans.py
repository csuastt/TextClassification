
# 所有需要修改的文件
file_paths = [
    "./data/ISEAR ID",
    "./data/ISEAR ID_test",
    "./data/ISEAR ID_train",
    "./data/ISEAR ID_validation"
]


def not_update(file):
    '''
    不修改数据, 只是换一种格式
    '''
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        j = 0
        line_list = list(f)
        for line in line_list:
            j += 1
            print(j, len(line_list))
            new_line = ""
            word_list = line.split()
            for i in range(len(word_list)):
                if i >= 9:
                    if i != len(word_list) - 1:
                        new_line = new_line + word_list[i] + ' '
                    else:
                        new_line = new_line + word_list[i]
                else:
                    new_line = new_line + word_list[i] + '\t'
            file_data += new_line + '\n'

    with open(file + "_b", "w" , encoding="utf-8") as f:
        f.write(file_data)

for file in file_paths:
    not_update(file)