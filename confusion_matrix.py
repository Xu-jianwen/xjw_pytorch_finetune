import matplotlib.pyplot as plt


def Confusion_Matrix(num_classes, label, matrix):

    print(matrix)
    plt.imshow(matrix, cmap=plt.cm.Blues)

    # 设置x轴坐标label
    plt.xticks(range(num_classes), label)
    # 设置y轴坐标label
    plt.yticks(range(num_classes), label)
    # 显示colorbar
    # plt.colorbar()
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.title("Confusion matrix")

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(matrix[y, x])
            plt.text(
                x,
                y,
                info,
                verticalalignment="center",
                horizontalalignment="center",
                color="white" if info > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
