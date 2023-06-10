from model import Model
import privateModule

if __name__ == '__main__':
    myModel = Model()

    myModel.Learn(privateModule.path1, privateModule.path2)

    del myModel
