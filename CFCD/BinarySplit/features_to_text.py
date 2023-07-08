# 将提取到的深度特征写入feature.txt
def feature_text(feature_list, path_features):
    with open('%sfeature.txt' % path_features, 'w', encoding='utf-8') as f:
        for i in range(len(feature_list)):
            feat = feature_list[i][0]
            for j in range(len(feat)):
                f.write(str(feat[j]))
                if j != len(feat) - 1:
                    f.write(',')
            f.write('\n')
